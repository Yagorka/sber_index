"""Аудит пользовательских выгрузок СберИндекса; исходные файлы не изменяются."""

import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from src.municipal import sha256


def dataset_group(filename):
    if filename.startswith("potrebitelskaya-aktivnost"):
        return "activity"
    if filename.startswith("ver-izmenenie-trat"):
        return "weekly_growth"
    if filename.startswith("consumer-spending-growth"):
        return "monthly_growth"
    if filename.startswith("consumper-spending-index-sa"):
        return "monthly_real_index"
    if filename.startswith("consumer-spending_ru"):
        return "monthly_spending"
    raise ValueError(f"Неизвестный набор: {filename}")


def read_export(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path), None
    if path.name.endswith(".csv.zip"):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Ожидался один CSV внутри {path.name}")
            with archive.open(members[0]) as file:
                return pd.read_csv(file, sep=";", encoding="utf-8-sig"), members[0]
    return pd.read_csv(path, sep=";", encoding="utf-8-sig"), None


def canonical_observations(raw, group):
    df = raw.copy()
    df = df.rename(columns={"date": "period"})
    if group == "activity":
        if "age" not in df:
            split = df.category.str.rsplit(",", n=1, expand=True)
            if split.shape[1] != 2:
                raise ValueError("Не удалось выделить возрастную группу")
            df["ipa_type"], df["age"] = split[0], split[1]
        dimensions = ["ipa_type", "age"]
    elif group == "weekly_growth":
        dimensions = ["category"]
    elif group == "monthly_growth":
        dimensions = ["type", "value_type"]
    else:
        dimensions = ["type"]
    normalized = df[dimensions].astype("string").apply(lambda c: c.str.strip())
    if normalized.isna().any().any():
        raise ValueError("Пустой ключ ряда")
    series = normalized.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True), axis=1)
    result = pd.DataFrame({"period": pd.to_datetime(df.period, errors="raise"),
                           "series_id": series, "value": pd.to_numeric(df.value, errors="raise")})
    return result.sort_values(["series_id", "period"]).reset_index(drop=True), dimensions


def compare_observations(left, right, atol=1e-10):
    key = ["period", "series_id"]
    if left.duplicated(key).any() or right.duplicated(key).any():
        return {"same_keys": False, "same_values_with_tolerance": False,
                "max_abs_difference": None, "comparison_status": "duplicate_keys"}
    merged = left.merge(right, on=key, how="outer", suffixes=("_left", "_right"),
                        indicator=True, validate="one_to_one")
    common = merged._merge == "both"
    diff = (merged.loc[common, "value_left"] - merged.loc[common, "value_right"]).abs()
    same_keys = bool(common.all())
    finite = np.isfinite(merged.loc[common, ["value_left", "value_right"]].to_numpy()).all()
    return {"same_keys": same_keys,
            "same_values_with_tolerance": bool(same_keys and finite and len(diff) > 0 and (diff <= atol).all()),
            "max_abs_difference": float(diff.max()) if len(diff) else None,
            "n_common": int(common.sum()), "n_left_only": int((merged._merge == "left_only").sum()),
            "n_right_only": int((merged._merge == "right_only").sum()),
            "comparison_status": "compared"}


def audit_exports(root: Path):
    paths = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.name.endswith(".csv") or p.name.endswith(".csv.zip") or p.name.endswith(".parquet")))
    raw_tables, tables, records, series_records = {}, {}, [], []
    for path in paths:
        group = dataset_group(path.name)
        raw, member = read_export(path)
        table, dimensions = canonical_observations(raw, group)
        raw_tables[path.name], tables[path.name] = raw, table
        frequency = "weekly" if group in ["activity", "weekly_growth"] else "monthly"
        key = ["period", "series_id"]
        conflicts = table.groupby(key).value.nunique(dropna=False).gt(1).sum()
        records.append({"file": path.name, "group": group, "rows": len(raw), "columns": len(raw.columns),
                        "series": table.series_id.nunique(), "dates": table.period.nunique(),
                        "start": table.period.min().date().isoformat(), "end": table.period.max().date().isoformat(),
                        "frequency": frequency, "unit": " | ".join(map(str, raw.unit_measure.unique())) if "unit_measure" in raw else "не указана в упрощённом CSV",
                        "geography": " | ".join(map(str, raw.ref_area.unique())) if "ref_area" in raw else "поле географии отсутствует",
                        "missing_cells": int(raw.isna().sum().sum()), "exact_duplicate_rows": int(raw.duplicated().sum()),
                        "duplicate_keys": int(table.duplicated(key).sum()), "conflicting_keys": int(conflicts),
                        "nonfinite_values": int((~np.isfinite(table.value)).sum()),
                        "negative_values": int((table.value < 0).sum()),
                        "min_value": float(table.value.min()), "max_value": float(table.value.max()),
                        "has_municipal_id": bool(set(raw.columns) & {"territory_id", "municipal_id", "oktmo"}),
                        "dimensions": " | ".join(dimensions), "bytes": path.stat().st_size,
                        "sha256": sha256(path), "archive_member": member})
        for series_id, current in table.groupby("series_id"):
            expected = pd.date_range(current.period.min(), current.period.max(),
                                     freq="7D" if frequency == "weekly" else "MS")
            observed = pd.DatetimeIndex(current.period.unique())
            series_records.append({"file": path.name, "group": group, "series_id": series_id,
                                   "n_observations": len(current), "missing_periods": len(expected.difference(observed)),
                                   "off_grid_periods": len(observed.difference(expected)),
                                   "start": current.period.min().date().isoformat(),
                                   "end": current.period.max().date().isoformat()})
    inventory = pd.DataFrame(records)
    comparisons = []
    for group, files in inventory.groupby("group"):
        for left, right in itertools.combinations(files.file, 2):
            result = compare_observations(tables[left], tables[right])
            full_equal = False
            if set(raw_tables[left].columns) == set(raw_tables[right].columns):
                # Сравнение метаданных с независимостью от порядка строк и типов CSV/Parquet.
                a = raw_tables[left].copy()
                b = raw_tables[right].copy()
                cols = sorted(set(a.columns) - {"value"})
                for col in cols:
                    a[col] = a[col].astype("string")
                    b[col] = b[col].astype("string")
                a, b = a.sort_values(cols).reset_index(drop=True), b.sort_values(cols).reset_index(drop=True)
                full_equal = bool(a[cols].equals(b[cols]) and len(a) == len(b) and
                                  np.allclose(a.value, b.value, rtol=0, atol=1e-10))
            comparisons.append({"group": group, "left": left, "right": right,
                                "same_file_bytes": files.set_index("file").loc[left, "sha256"] == files.set_index("file").loc[right, "sha256"],
                                "same_full_table_with_tolerance": full_equal, **result})
    return inventory, pd.DataFrame(comparisons), pd.DataFrame(series_records), raw_tables, tables
