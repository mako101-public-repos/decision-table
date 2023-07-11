import pytest
from pathlib import Path
from app.models.decision_data_holder import DecisionDataHolder
from app.models.decision_table import DecisionTable


@pytest.fixture
def decision_table_nr1():
    return DecisionTable.create_from_csv(
        Path("tests/resources/decision_tables/scoring_process_result.csv")
    )


def test_rejected(decision_table_nr1):
    ddh = DecisionDataHolder({
        "hard_check_passed": False,
        "risk_score": 8,
        "all_data_collected": True
    })
    decision_table_nr1.evaluate(ddh)
    assert "status" in ddh
    assert ddh["status"] == "REJECTED"

    ddh["all_data_collected"] = False
    decision_table_nr1.evaluate(ddh)
    assert "status" in ddh
    assert ddh["status"] == "REJECTED"


def test_more_data(decision_table_nr1):
    ddh = DecisionDataHolder({
        "hard_check_passed": True,
        "risk_score": 8,
        "all_data_collected": False
    })
    decision_table_nr1.evaluate(ddh)
    assert "status" in ddh
    assert ddh["status"] == "MORE_DATA"


def test_approved(decision_table_nr1):
    ddh = DecisionDataHolder({
        "hard_check_passed": True,
        "risk_score": 12,
        "all_data_collected": True
    })
    decision_table_nr1.evaluate(ddh)
    assert "status" in ddh
    assert ddh["status"] == "APPROVED"


def test_no_row_matched(decision_table_nr1):
    ddh = DecisionDataHolder({
        "hard_check_passed": True,
        "risk_score": 9,
        "all_data_collected": True
    })
    decision_table_nr1.evaluate(ddh)
    assert "status" not in ddh
