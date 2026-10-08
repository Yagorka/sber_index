"""Восстановить авторские небольшие входы без перезаписи локальных данных."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore_references(root=ROOT):
    catalog = json.loads((root / "configs/data_provenance.json").read_text())
    destination = root / "data/inputs"
    destination.mkdir(parents=True, exist_ok=True)
    restored = []
    for entry in catalog["files"]:
        if not entry.get("reference_path"):
            continue
        source = root / entry["reference_path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Изменён reference-файл: {source}")
        target = destination / entry["name"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Локальный файл отличается, перезапись запрещена: {target}")
        else:
            shutil.copyfile(source, target)
            restored.append(entry["name"])
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    if args.references:
        print("Восстановлены:", ", ".join(restore_references()) or "все reference-файлы уже на месте")
    else:
        catalog = json.loads((ROOT / "configs/data_provenance.json").read_text())
        for entry in catalog["files"]:
            print(f"{entry['name']}: {entry['kind']} — {entry['source']}")
        print("Скачиваемые файлы сохраняйте в data/inputs/; точный снимок проверяется по SHA256.")


if __name__ == "__main__":
    main()
