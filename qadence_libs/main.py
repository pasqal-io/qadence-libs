from __future__ import annotations

from typing import Optional


def main(str_to_add: Optional[str] = None) -> str:
    msg = "Pasqal template Python project"
    if str_to_add is not None:
        msg += str_to_add
    return msg


if __name__ == "__main__":
    msg = main(str_to_add="from same file")
    print(msg)
