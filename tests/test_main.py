from __future__ import annotations

from template_python.main import main

expected_msg = "Pasqal template Python project"


def test_main() -> None:
    msg = main()
    assert msg == expected_msg


def test_main_with_str() -> None:
    str_to_add = "with added str"
    msg = main(str_to_add=str_to_add)
    assert msg == expected_msg + str_to_add
