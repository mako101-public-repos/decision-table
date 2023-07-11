import re
import operator as o
import pandas as pd
from pathlib import Path
from distutils.util import strtobool
from typing import Union

from app.models.abstract import AbstractDecisionTable
from app.models.decision_data_holder import DecisionDataHolder


class DecisionTable(AbstractDecisionTable):

    """
    This is a mapping of comparator functions we will be using later
    to evaluate the values in the DDH against the conditions in the decision table
    Can be easily extended with the additional operators if needed
    """

    OPS_MAP = {
        ">": o.gt,
        "<": o.lt,
        ">=": o.ge,
        "<=": o.le,
        "=": o.eq
    }

    def __init__(self, decision_table: pd.DataFrame = None) -> None:
        self.decision_table = decision_table

    @staticmethod
    def parse_cell_value(value: str) -> tuple:
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
            if not any(operator == key for key in DecisionTable.OPS_MAP.keys()):
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

    @staticmethod
    def create_from_csv(filepath: Path) -> "DecisionTable":
        """
        This method will ingest the decision table CSV file and convert it into a table, presented as a Pandas DataFrame
        It will split the ingested string values into an operator and a comparison value
        and store the resultant tuples as values of the DataFrame cells
        the CSV file may contain any number of columns and rows and use any desired column names

        :param: filepath: Posix filepath to the CSV file containing the decision table
        :returns: DecisionTable() class instance with the decision table
        """

        # firstly let us verify whether the given file path is relative or absolute
        # if it is absolute, we will take it as-is
        # if it is relative, then we assume that any file specified will be relative to the root directory of this program
        # and append the root directory to the given filepath

        if not filepath.is_absolute():

            base_path = Path(__file__).parent.parent.parent.resolve()
            filepath = base_path.joinpath(filepath)

        # we will use pandas library to create a table (presented as a DataFrame object) from CSV
        data = pd.read_csv(filepath, sep=";")

        # Now we apply parse_cell_value() function to all the cells in the DataFrame
        # to convert the strings they currently contain into objects we will later use in the `evaluate` method
        data = data.applymap(DecisionTable.parse_cell_value)
        dt = DecisionTable(decision_table=data)
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
        # firstly, lets check if a decision table has been imported
        if self.decision_table is None:
            raise ValueError('Please import a decision table first by running `DecisionTable.create_from_csv()`')

        # secondly, lets check that the imported decision table contains the `status` column
        status_column = 'status'
        if status_column not in self.decision_table.columns:
            raise ValueError(f'{status_column} column not found in the decision table')

        # thirdly, if the current DDH has already been evaluated, it may already contain the `status` key
        # this may produce an incorrect evaluation, in case where there is no match, and the existing result is not overriden
        # so let's delete it
        if status_column in ddh.data.keys():
            del ddh[status_column]

        for row_no, row in self.decision_table.iterrows():
            for ddh_key in ddh.data.keys():
                try:
                    operator, value = row[ddh_key]
                except KeyError:
                    raise ValueError(f'Could not find column {ddh_key} in the decision table, unable to evaluate')
                if not operator:
                    raise ValueError(f'Decision table row {row_no + 1}: '
                                     f'Operator not specified for column {ddh_key}, value {value}')
                op_func = self.OPS_MAP[operator]
                # break out of the loop anytime there is no match
                # move to the next row in the table
                match = op_func(ddh.data[ddh_key], value)
                if not match:
                    break

            # if we get here then the current row is a match
            # insert the status value from this row into the DDH object
            else:
                ddh[status_column] = row[status_column][1]
                return True
        return False
