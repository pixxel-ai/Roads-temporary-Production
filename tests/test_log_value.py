import sys
sys.path.append('../')
import pytest
from generate_predictions import log_value
import logging

test_values = (['anInt', 123],
               ['word', 'hello'],
               ['aFloat', 3.4],
               ['aBoolean', True],
               ['anObject', object()])

@pytest.mark.parametrize('test_value', test_values)
def test_log_value(test_value):
    create_var_and_test(test_value[0], test_value[1])

def create_var_and_test(name, value):
    expected = name + ' : ' + str(value) # Expected return value
    assert log_value(name, value, logging) == expected