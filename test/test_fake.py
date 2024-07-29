import pytest
from src import foo

def test_foo(capfd):
    # Capture the output
    foo()
    captured = capfd.readouterr()
    # Assert that the expected output is printed
    assert 'HELLO!!' in captured.out

if __name__ == '__main__':
    pytest.main()