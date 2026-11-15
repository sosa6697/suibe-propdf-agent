"""允许使用 `python -m campus_rules ...` 调用 CLI。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
