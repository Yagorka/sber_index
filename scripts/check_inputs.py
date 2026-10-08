"""Проверяет наличие локальных входов; не скачивает данные и не запускает модели."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((ROOT / "configs/input_files.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=catalog["profiles"], default="full")
    args = parser.parse_args()
    missing = []
    checked = 0
    for entry in catalog["files"]:
        required = entry["required_for"]
        if not required or (args.profile != "full" and args.profile not in required):
            continue
        path = ROOT / catalog["directory"] / entry["name"]
        checked += 1
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path.relative_to(ROOT).as_posix())
    if missing:
        print("Не хватает локальных файлов:\n" + "\n".join(missing))
        print("Поместите их в data/inputs/. Список и назначение: docs/INPUT_DATA_RU.md")
        return 1
    print(f"Профиль {args.profile}: все {checked} обязательных файлов присутствуют в data/inputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
