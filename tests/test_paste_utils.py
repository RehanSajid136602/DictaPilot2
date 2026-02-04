from paste_utils import compute_delta, longest_common_prefix


def test_longest_common_prefix():
    assert longest_common_prefix("abc", "abd") == 2
    assert longest_common_prefix("same", "same") == 4
    assert longest_common_prefix("", "x") == 0


def test_delta_calc():
    to_delete, to_insert = compute_delta("hello world", "hello there")
    assert to_delete == 5
    assert to_insert == "there"

    to_delete, to_insert = compute_delta("hello", "hello world")
    assert to_delete == 0
    assert to_insert == " world"

    to_delete, to_insert = compute_delta("abc", "")
    assert to_delete == 3
    assert to_insert == ""
