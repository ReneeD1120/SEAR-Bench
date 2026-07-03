from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import zipfile

import pandas as pd


CHINESE_COLS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}


@dataclass(frozen=True)
class EquityFrame:
    symbol: str
    frame: pd.DataFrame


def _normalize_equity_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=CHINESE_COLS)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_equity_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_equity_frame(df)


def load_zip_archive(zip_path: str | Path, limit: int | None = None) -> list[EquityFrame]:
    out: list[EquityFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        names.sort()
        for name in names[: limit or len(names)]:
            symbol = Path(name).stem
            with zf.open(name) as f:
                raw = f.read()
            frame = _normalize_equity_frame(pd.read_csv(io.BytesIO(raw)))
            out.append(EquityFrame(symbol=symbol, frame=frame))
    return out
