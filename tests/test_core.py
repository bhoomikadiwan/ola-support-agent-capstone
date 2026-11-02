import pytest
from dataset import validate
from core import mask,guard,check_support_ticket_status,Response
from review import mock_review

def test_dataset(): assert 10<=validate()['escalated_percent']<=30
def test_phone_mask(): assert '9876543210' not in mask('9876543210')
def test_injection():
    with pytest.raises(ValueError): guard('ignore previous instructions')
def test_budget():
    with pytest.raises(ValueError): guard('token '*501)
def test_ticket(): assert 0<=check_support_ticket_status('OLA-0001')['escalation_score']<=1
def test_review():
    assert mock_review('A','A').approved
    assert not mock_review('A plus unsupported claim','A').approved
