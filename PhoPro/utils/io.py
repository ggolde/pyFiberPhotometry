import json
import os

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


def _natsorted(values: pd.Index) -> list:
    try:
        from natsort import natsorted
    except ImportError:
        return sorted(values)
    return natsorted(values)


def _is_missing_scalar(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if not np.isscalar(value):
        return False
    return bool(pd.isna(value))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    return str(value)


def _json_dumps_or_na(value):
    if _is_missing_scalar(value):
        return pd.NA
    return json.dumps(value, default=_json_default)


######################
#region --- OUTPUT ---
######################
def append_to_csv(csv_path: str, df: pd.DataFrame) -> None:
    """
    Append rows of df to an existing .csv file at csv_path or creates it if it does not exist.
    Args:
        csv_path (str) : Path to csv file.
        df (pd.DataFrame) : DataFrame to append to csv.
    Returns:
        None
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, mode="w")
    else:
        # columns must match existing header
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if list(df.columns) != existing_cols:
            raise ValueError(
                f"CSV column mismatch.\nExisting: {existing_cols}\nIncoming: {list(df.columns)}"
            )
        df.to_csv(csv_path, index=False, mode="a", header=False)

def write_dict_to_txt(d: dict, filename: str, indent: int = 0) -> None:
    """
    Write a nested dictionary to a human-readable, indented text file.
    Args:
        d (dict): Dictionary to serialize. Nested dictionaries are supported.
        filename (str): Path to the output text file.
        indent (int, optional): Initial indentation level (in tabs).
    Returns:
        None
    """
    def _write_dict(fd, obj, lvl):
        for k, v in obj.items():
            if isinstance(v, dict):
                fd.write("\t" * lvl + f"{k}:\n")
                _write_dict(fd, v, lvl + 1)
            else:
                fd.write("\t" * lvl + f"{k}: {v}\n")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        _write_dict(f, d, indent)

#endregion

def clean_axis_dtype(
        df: pd.DataFrame,
        convert_strings_to_categoricals: bool = True
        ) -> pd.DataFrame:
    """Return a copy with axis metadata dtypes normalized for AnnData writing."""
    out = df.copy().infer_objects()
    out.index = out.index.astype(str)

    for col in out.columns:
        values = out[col]._values
        if isinstance(values, pd.arrays.FloatingArray):
            out[col] = out[col].astype(float)

    if convert_strings_to_categoricals:
        for col in out.columns:
            if infer_dtype(out[col]) != "string":
                continue

            categories = pd.Categorical(out[col])
            if len(categories.categories) >= len(categories):
                continue

            out[col] = categories

    for col in out.columns:
        if out[col].dtype != object or infer_dtype(out[col]) == "string":
            continue
        out[col] = out[col].map(_json_dumps_or_na).astype("string")

    return out

#####################
#region --- INPUT ---
#####################
def read_txt_to_dict(filename: str) -> dict:
    """
    Read a nested dictionary from a text file created by ``write_dict_to_txt``.
    Args:
        filename (str): Path to the input text file.
    Returns:
        dict: Nested dictionary reconstructed from the file contents
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    def parse(lines, start=0, level=0):
        out = {}
        i = start
        while i < len(lines):
            raw = lines[i]
            curr = len(raw) - len(raw.lstrip("\t"))
            if curr < level:
                break
            s = raw.strip()
            i += 1
            if not s:
                continue
            if s.endswith(":"):  # nested block
                key = s[:-1]
                sub, nxt = parse(lines, i, level + 1)
                out[key] = sub
                i = nxt
            else:
                if ":" in s:
                    k, v = s.split(":", 1)
                    out[k.strip()] = v.strip()
        return out, i

    d, _ = parse(lines, 0, 0)
    return d

#endregion
