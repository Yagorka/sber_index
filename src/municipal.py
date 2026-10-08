"""Чтение СберИндекса без изменения исходных файлов. Интервалы: [from, to)."""

from pathlib import Path
import hashlib
import sqlite3
import struct

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dictionary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"oktmo": "string", "shape_linked_oktmo": "string"})
    for column in ["territory_id", "region_code", "year_from", "year_to", "shape"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("Int64")
    df["territory_id"] = df["territory_id"].astype("string")
    return df


def gpkg_header(blob):
    """Проверить заголовок GeoPackage и извлечь envelope; это НЕ проверка топологии."""
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        return {"valid_header": False, "empty": None, "xmin": None, "xmax": None,
                "ymin": None, "ymax": None, "srs_id": None}
    flags = blob[3]
    endian = "<" if flags & 1 else ">"
    envelope_type = (flags >> 1) & 7
    lengths = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}
    n = lengths.get(envelope_type)
    valid = n is not None and len(blob) > 8 + 8 * n and blob[2] == 0
    bounds = (None,) * 4
    if valid and n:
        bounds = struct.unpack(endian + "d" * n, blob[8:8 + 8 * n])[:4]
    return dict(zip(["valid_header", "empty", "xmin", "xmax", "ymin", "ymax", "srs_id"],
                    [valid, bool(flags & 16), *bounds,
                     struct.unpack(endian + "i", blob[4:8])[0]]))


def load_geometry_metadata(path: Path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        contents = pd.read_sql_query("SELECT * FROM gpkg_contents", con)
        spatial = pd.read_sql_query("SELECT * FROM gpkg_geometry_columns", con)
        # Таблица известна из структуры этого набора; SQL не принимает внешний ввод.
        df = pd.read_sql_query(
            "SELECT fid, territory_id, year_from, year_to, osm_ref, osm_vers, geom "
            "FROM t_dict_municipal_districts_poly", con)
    header = pd.DataFrame([gpkg_header(blob) for blob in df.pop("geom")])
    df["territory_id"] = df["territory_id"].astype("string")
    return pd.concat([df, header], axis=1), contents, spatial, integrity


def active_in_year(df, year: int):
    return df.loc[(df.year_from <= year) & (year < df.year_to)].copy()


def interval_issues(df: pd.DataFrame, key: str):
    """Возвращает все пары перекрытий и разрывы соседних версий одного ключа."""
    records = []
    for identity, group in df.groupby(key, dropna=False):
        values = group.sort_values(["year_from", "year_to"]).to_dict("records")
        for left, right in zip(values, values[1:]):
            if left["year_to"] < right["year_from"]:
                records.append({"key": identity, "issue": "gap", "left_from": left["year_from"],
                                "left_to": left["year_to"], "right_from": right["year_from"]})
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                if max(left["year_from"], right["year_from"]) < min(left["year_to"], right["year_to"]):
                    records.append({"key": identity, "issue": "overlap",
                                    "left_from": left["year_from"], "left_to": left["year_to"],
                                    "right_from": right["year_from"]})
    return pd.DataFrame(records, columns=["key", "issue", "left_from", "left_to", "right_from"])


def attribute_transitions(df):
    records = []
    for identity, group in df.groupby("territory_id"):
        values = group.sort_values("year_from").to_dict("records")
        for before, after in zip(values, values[1:]):
            before_status = before["municipal_district_status"]
            after_status = after["municipal_district_status"]
            before_status = None if pd.isna(before_status) else before_status
            after_status = None if pd.isna(after_status) else after_status
            records.append({"territory_id": identity, "year": after["year_from"],
                            "region_name": after["region_name"],
                            "name_before": before["municipal_district_name"],
                            "name_after": after["municipal_district_name"],
                            "type_before": before["municipal_district_type"],
                            "type_after": after["municipal_district_type"],
                            "oktmo_before": before["oktmo"], "oktmo_after": after["oktmo"],
                            "status_before": before_status, "status_after": after_status,
                            "status_changed": before_status != after_status,
                            "name_changed": before["municipal_district_name"] != after["municipal_district_name"],
                            "type_changed": before["municipal_district_type"] != after["municipal_district_type"],
                            "oktmo_changed": before["oktmo"] != after["oktmo"]})
    return pd.DataFrame(records)


def transformation_events(df):
    """Группировка по change_id: одна трансформация может связывать несколько МО."""
    incoming = df.dropna(subset=["change_id_from"])
    outgoing = df.dropna(subset=["change_id_to"])
    ids = sorted(set(incoming.change_id_from) | set(outgoing.change_id_to))
    result = []
    for change in ids:
        before = outgoing[outgoing.change_id_to == change]
        after = incoming[incoming.change_id_from == change]
        years_before = sorted(before.year_to.unique().tolist())
        years_after = sorted(after.year_from.unique().tolist())
        result.append({"change_id": change, "kind": change.rsplit("_", 1)[0],
                       "n_before": len(before), "n_after": len(after),
                       "territories_before": "|".join(sorted(set(before.territory_id))),
                       "territories_after": "|".join(sorted(set(after.territory_id))),
                       "years_before": str(years_before), "years_after": str(years_after),
                       "both_sides_present": len(before) > 0 and len(after) > 0,
                       "years_match": years_before == years_after})
    return pd.DataFrame(result)
