import re
import csv
import operator as o
from pathlib import Path
from distutils.util import strtobool
from typing import Union, List

from app.models.abstract import AbstractDecisionTable
from app.models.decision_data_holder import DecisionDataHolder
from app.exceptions import MissingDataError, MissingColumnError


class DecisionTable(AbstractDecisionTable):
    """
    A class designed to ingest decision tables from CSV files and evaluate DecisionDataHolder objects

    The expected usage is as follows:
    1. A class instance is created by calling `DecisionTable.create_from_csv()` static method
    with the filepath of the CSV file
    2. The class instance will automatically receive the CSV data and build a decision matrix from it
    3. The class instance is now ready to assess DecisionDataHolder objects via .evaluate() method
    """

    # This is a mapping of comparator functions we will be using later
    # to evaluate the values in the DDH against the conditions in the decision table
    # Can be easily extended with the additional operators if needed
    _OPS_MAP = {
        ">": o.gt,
        "<": o.lt,
        ">=": o.ge,
        "<=": o.le,
        "=": o.eq
    }

    def __init__(self) -> None:
        self._csv_data: List[dict] = list()
        self._decision_matrix: List[dict] = list()

    @property
    def csv_data_loaded(self) -> bool:
        """Checks if the CSV data has been loaded"""
        return len(self._csv_data) != 0

    @property
    def decision_matrix_built(self) -> bool:
        """Checks if the decision matrix has been built"""
        return len(self._decision_matrix) != 0

    def __parse_cell_value(self, value: str) -> tuple:
        """
        This method is used to parse the cell value from the ingested decision table
        and split it into a comparison operator (if present) and an integer or a boolean value
        The supported operators are limited to those found in DecisionTable.OPS_MAP mapping
        :param: value string
        :returns: (operator, value)
        """
        operator_matched = re.search(r"[><=]{1,2}", value)
        # if nothing is matched, assume the value is an alphanumeric string and return it as-is
        if not operator_matched:
            return None, value
        else:
            operator = operator_matched.group(0)
            # Do a quick sanity check and verify whether the extracted operator is a known one,
            # ie present in the DecisionTable.OPS_MAP mapping. Raise an error if nothing is matched
            if not any(operator == key for key in self._OPS_MAP.keys()):
                raise ValueError(f'Unsupported operator {operator}')

            # Now let's analyse the remaining string and work out if it is an integer or a boolean
            # those are the currently known data types, support for more can be added if needed
            # raise an error if its neither

            final_value: Union[bool, int]
            # remove the operator from the value string
            rem_value = re.sub(str(operator), "", value)
            if rem_value.lower() in ['true', 'false']:
                final_value = bool(strtobool(rem_value))
            # the minus needs to be removed before testing if a value is a digit
            elif rem_value.isdigit() or (rem_value.startswith('-') and rem_value[1:].isdigit()):
                final_value = int(rem_value)
            else:
                raise ValueError(f'Unknown value type {rem_value}')

            return operator, final_value

    def __parse_csv_data(self) -> None:
        """
        Internal method to parse the values of all decision row dictionaries
        Builds up the internal decision matrix, represented as a list of dictionaries
        """
        if self.csv_data_loaded:
            for plaintext_dict in self._csv_data:
                self._decision_matrix.append({
                    column: self.__parse_cell_value(value)
                    for column, value in plaintext_dict.items()
                })
        else:
            raise MissingDataError('Please import CSV data first by running `DecisionTable.create_from_csv()`')

    def load_csv_data(self, csv_data: List[dict]) -> None:
        """
        A setter method for transmuted CSV data
        Triggers internal method to parse it further
        """
        self._csv_data = csv_data
        self.__parse_csv_data()

    @staticmethod
    def create_from_csv(filepath: Path) -> "DecisionTable":
        """
        This method ingests the decision table CSV file and represents its rows as a list of dictionaries
        It then creates an instance of DecisionTable() class and passes the CSV data
        to the instance's internal methods for further processing

        :param: filepath: Posix filepath to the CSV file containing the decision table
        :returns: DecisionTable() class instance with the decision table loaded and ready to use
        """

        # firstly let us verify whether the given file path is relative or absolute
        # if it is absolute, we will take it as-is
        # if it is relative, then we assume that any file specified will be relative to the root directory of this program
        # and append the root directory to the given filepath

        if not filepath.is_absolute():
            base_path = Path(__file__).parent.parent.parent.resolve()
            filepath = base_path.joinpath(filepath)

        # now we are going to read the CSV file
        # we will take the list of rows (returned as lists of strings by the CSV reader)
        # and transmute them into a list of dictionaries with row headers as keys
        # for now we leave cell values as strings, they will be parsed later
        csv_raw, csv_data = list(), list()
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            for row in csv_reader:
                csv_raw.append(row)

            headers = csv_raw.pop(0)
            for row in csv_raw:
                row_dict = dict(zip(headers, row))
                csv_data.append(row_dict)

        # now we instantiate the DecisionTable() class
        # and pass it the list of dictionaries from above for further processing
        dt = DecisionTable()
        dt.load_csv_data(csv_data)
        return dt

    def evaluate(self, ddh: DecisionDataHolder) -> bool:
        """
        This method will evaluate the values of predictors against the conditions in the decision table
        and retrieve a relevant status from the table (if all conditions are matched)
        The following conditions need to be satisfied before using this method:
            - the decision table needs to be imported
            - the decision table should have the `status` column that will contain the final decision values
        There are checks to verify both

        :param ddh: DecisionDataHolder object containing predictors as `key: value` pairs
        :returns: True/False boolean
        """
        # Check if a decision table has been imported
        if not self.decision_matrix_built:
            raise MissingDataError('Please instantiate and populate the class instance '
                                   'by running `DecisionTable.create_from_csv()`')

        # If the current DDH has already been evaluated, it may already contain the `status` key
        # This may produce an incorrect evaluation, in case where there is no match, and the existing result is not overriden
        # Therefore it needs to be removed, if present
        status_column = 'status'
        if status_column in ddh.data.keys():
            del ddh[status_column]

        # Check that the decision row dictionary contains the `status` key
        for decision_dict in self._decision_matrix:
            if status_column not in decision_dict.keys():
                raise MissingColumnError(f'{status_column} column not found in decision row: {decision_dict}')

            for ddh_key in ddh.data.keys():
                try:
                    operator, value = decision_dict[ddh_key]
                except KeyError:
                    raise ValueError(f'Could not find column {ddh_key} in the decision table, unable to evaluate')
                if not operator:
                    raise ValueError(f'Decision row {decision_dict}: '
                                     f'Operator not specified for column {ddh_key}, value {value}')
                op_func = self._OPS_MAP[operator]
                # break out of the loop any time there is no match
                # move to the next row in the table
                match = op_func(ddh.data[ddh_key], value)
                if not match:
                    break

            # if we get here then the current row is a match
            # insert the status value from this row into the DDH object
            else:
                ddh[status_column] = decision_dict[status_column][1]
                return True
        return False
