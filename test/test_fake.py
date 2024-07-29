import pytest

def test_fake():
    # Replace this with your test logic
    assert True

@pytest.mark.parametrize("input_value, expected_result", [
    # Add test cases here
    (1, 1),
    (2, 2),
    (3, 3),
])
def test_fake_parametrized(input_value, expected_result):
    # Replace this with your test logic
    assert input_value == expected_result

if __name__ == '__main__':
    pytest.main()