Interview task Decision Table
=============================
The aim of the task is to write the evaluation of the decision table according to the following description, including that
the solution satisfies all supplied unit tests. The solution should be well extensible and easy to read.

Decision table
--------------
Decision table contains input and output columns. Input columns are to the left of the column with asterisk.
Outputs are to the right of the column with asterisk. The column name is the name of the input or output predictor.
The input cells always specify a condition that relates to the input predictor.

+ ----------------- + ---------- + ------------------ + - +------------ +
| hard_check_passed | risk_score | all_data_collected | * |   status    |
+ ----------------- + ---------- + ------------------ + - +------------ +
| =false            | >-99999    | =true              | * | "REJECTED"  |
| =false            | >-99999    | =false             | * | "REJECTED"  |
| =true             | >-99999    | =false             | * | "MORE_DATA" |
| =true             | >10        | =true              | * | "APPROVED"  |
| =false            | <=10       | =true              | * | "REJECTED"  |
+ ----------------- + ---------- + ------------------ + - +------------ +

Evaluation of table
-------------------
The values of predictors needed for evaluation the table are after filling defined in the DecisionDataHolder object in file /app/models/decision_data_holder.py.
The table is evaluated by traversing the table row by row, from the first to the last. All input conditions are tested on each row.
If all conditions on the row are met, all output cells (values) are written to the output predictors. By finding the row
on which all conditions are met and writing the output predictors to the DecisionDataHolder, the whole evaluation of the table ends.

Example of evaluation
---------------------
DecisionDataHolder contains predictors:
{
    "hard_check_passed": False,
    "risk_score": 8,
    "all_data_collected" False
}

1) For the first row full condition is (hard_check_passed is False) and (risk_score > -99999) and (all_data_collected is True).
   Condition is not met. So we continue with the next row.

2) For the second row full condition is  (hard_check_passed is False) and (risk_score > -99999) and (all_data_collected is False).
   Condition is met, so we write all output predictors to do DecisionDataHolder.
   ddh["status"] = "REJECTED"
   The evaluation ends here.

Source code
-----------
In this example, the decision table is defined in the CSV file /tests/resources/decision_tables/scoring_process_result.csv.
The file /app/models/decision_data_holder.py contains a DecisionDataHolder object that holds all defined predictors.
It is only a classic dictionary. There is an abstract class for DecisionTable in file /app/models/abstract.py that
defines the required methods.

The result of your work, the class
class DecisionTable(AbstractDecisionTable)
shall be placed into file /app/models/decision_table.py.
It is left up to your consideration to decompose the solution into more classes to achieve above stated objectives.

The project shall use Python 3.x, pipenv and pytest library for unit tests.
The produced source code should pass without warnings and errors through lint checks from bandit, flake8, mypy and black.
For security reasons, we do not want to use the eval() function or similar in the solution.
