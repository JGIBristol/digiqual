"""Reading simulation results from CSV files."""

import pandas as pd


def clean_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """
    Makes a raw results table safe to analyse: tidy, unique column names and numbers only.

    Column names are stripped, internal spaces become underscores, other special
    characters are removed and duplicates are numbered (``Signal``, ``Signal_1``).
    Columns with no numeric values are dropped; stray text cells in a numeric
    column become missing values, which validation later drops as invalid rows.

    Args:
        df (pd.DataFrame): The table as read from the file.

    Returns:
        tuple: The cleaned numeric DataFrame, the names of the dropped text
        columns, and the number of text cells turned into missing values.

    Examples:
        ```python
        df, dropped, n_bad = clean_columns(pd.read_csv("results.csv"))
        ```
    """
    df = df.copy()
    names = (df.columns.astype(str).str.strip()
             .str.replace(r'\s+', '_', regex=True)
             .str.replace(r'[^a-zA-Z0-9_]', '', regex=True))
    unique, seen = [], set()
    for name in names:
        new, counter = name, 1
        while new in seen:
            new = f"{name}_{counter}"
            counter += 1
        seen.add(new)
        unique.append(new)
    df.columns = unique

    numeric = df.apply(pd.to_numeric, errors="coerce")
    text_cols = [c for c in df.columns if numeric[c].isna().all() and df[c].notna().any()]
    n_bad_cells = int((numeric.isna() & df.notna()).drop(columns=text_cols).sum().sum())
    return numeric.drop(columns=text_cols), text_cols, n_bad_cells


def read_results_csv(path) -> pd.DataFrame:
    """Reads a results CSV and cleans it exactly as the DigiQual app does (see `clean_columns`)."""
    df, _text_cols, _n_bad = clean_columns(pd.read_csv(path))
    return df
