"""
Builds time-windowed features from the raw UNSW-NB15 capture.

Reads UNSW-NB15_1-4.csv, sorts by start time, and splits chronologically
into train and test. For each flow, computes 13 statistics over trailing
1s, 10s and 60s windows of prior flows from the same source IP:
arrival/byte/packet rate, mean and std of duration, mean TCP RTT, unique
destinations, unique destination ports, destination entropy, failed
connection ratio, burstiness, protocol count, and change in arrival rate.

Writes train_windowed.parquet and test_windowed.parquet.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

SUBSAMPLE_ROWS = None  # None for full ~2.54M; set for a timing test first
TEST_FRACTION = 0.2
WINDOWS = {"short": "1s", "medium": "10s", "long": "60s"}
FAILED_STATES = {"INT", "REQ"}

COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

RAW_DIR = find_dir("raw_data_full")
OUT_DIR = find_dir("data") / "temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_raw():
    parts = []
    for i in [1, 2, 3, 4]:
        df = pd.read_csv(RAW_DIR / f"UNSW-NB15_{i}.csv", header=None, names=COLUMNS,
                          low_memory=False, encoding="latin1")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if SUBSAMPLE_ROWS is not None and SUBSAMPLE_ROWS < len(df):
        df = df.sample(n=SUBSAMPLE_ROWS, random_state=42).reset_index(drop=True)
    return df

def _shannon_entropy(arr):
    """Shannon entropy of the value distribution in one rolling window."""
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def compute_window_features(df, seconds_str, prefix):
    """All rolling results are realigned to df's original row order before
    being returned. groupby().rolling() emits rows in group-sorted order
    (srcip, then time), NOT the caller's chronological order -- assigning
    .values directly silently attaches every window value to the wrong
    flow, which is a bug that is invisible in summary statistics."""
    win_s = pd.Timedelta(seconds_str).total_seconds()

    tmp = df[["srcip", "ts", "dur", "sbytes", "dbytes", "spkts", "dpkts", "tcprtt",
               "dstip", "dsport", "proto", "state"]].copy()
    tmp = tmp.reset_index(drop=True)  # integer index is the alignment key
    # rolling.apply only accepts numeric dtypes, so categorical columns are factorized first
    tmp["dstip_code"] = pd.factorize(tmp["dstip"])[0].astype(float)
    tmp["dsport_code"] = pd.factorize(tmp["dsport"])[0].astype(float)
    tmp["proto_code"] = pd.factorize(tmp["proto"])[0].astype(float)
    tmp["is_failed"] = tmp["state"].isin(FAILED_STATES).astype(float)
    tmp["byte_total"] = tmp["sbytes"] + tmp["dbytes"]
    tmp["pkt_total"] = tmp["spkts"] + tmp["dpkts"]
    # inter-arrival gap to the previous flow from the same source
    tmp["gap"] = tmp.groupby("srcip")["ts"].diff().dt.total_seconds().fillna(0.0)

    def roll_group(g):
        # g keeps the caller's integer index; gg is the same rows re-indexed by
        # time so pandas can do offset-based windowing. set_index preserves row
        # order, so .values from gg lines up positionally with g.index.
        gg = g.set_index("ts")
        r = gg.rolling(seconds_str)
        flow_count = r["dur"].count().values
        arrival_rate = flow_count / win_s
        gap_mean = r["gap"].mean().values
        gap_std = pd.Series(r["gap"].std().values).fillna(0.0).values
        return pd.DataFrame({
            f"{prefix}_arrival_rate": arrival_rate,
            f"{prefix}_byte_rate": r["byte_total"].sum().values / win_s,
            f"{prefix}_pkt_rate": r["pkt_total"].sum().values / win_s,
            f"{prefix}_mean_dur": r["dur"].mean().values,
            f"{prefix}_std_dur": pd.Series(r["dur"].std().values).fillna(0.0).values,
            f"{prefix}_mean_tcprtt": r["tcprtt"].mean().values,
            f"{prefix}_unique_dst": r["dstip_code"].apply(lambda x: np.unique(x).size, raw=True).values,
            f"{prefix}_unique_dst_port": r["dsport_code"].apply(lambda x: np.unique(x).size, raw=True).values,
            f"{prefix}_dst_entropy": r["dstip_code"].apply(_shannon_entropy, raw=True).values,
            f"{prefix}_failed_ratio": r["is_failed"].mean().values,
            # burstiness = coefficient of variation of inter-arrival gaps; a steady
            # stream sits near 0, a bursty one climbs. A single-flow window has no
            # variation to measure, so std is NaN there and 0 is the right reading.
            f"{prefix}_burstiness": gap_std / (gap_mean + 1e-9),
            f"{prefix}_proto_nunique": r["proto_code"].apply(lambda x: np.unique(x).size, raw=True).values,
            # spec's "changes in feature values relative to preceding windows"
            f"{prefix}_delta_arrival_rate": pd.Series(arrival_rate).diff().fillna(0.0).values,
        }, index=g.index)

    out = tmp.groupby("srcip", group_keys=False)[tmp.columns].apply(roll_group)
    return out.sort_index().reset_index(drop=True)

def main():
    print("=== loading raw UNSW-NB15 files ===")
    t0 = time.perf_counter()
    df = load_raw()
    print(f"loaded {len(df)} rows in {time.perf_counter()-t0:.1f}s")

    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["stime"], unit="s")
    df = df.sort_values("ts").reset_index(drop=True)

    # raw CSVs have inconsistent per-cell typing in several columns (e.g. sport/dsport
    # mixing decimal and hex-looking string port values) -- coerce explicitly so the
    # later parquet write doesn't choke on an object column with mixed Python types.
    nominal_cols = {"srcip", "dstip", "proto", "state", "service", "attack_cat"}
    for col in COLUMNS:
        if col in nominal_cols:
            df[col] = df[col].astype(str).str.strip()
        elif col not in ("stime", "ltime"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_time = df["ts"].iloc[split_idx]
    print(f"chronological split at row {split_idx} / {len(df)}, time={split_time}")

    for scale, seconds_str in WINDOWS.items():
        print(f"\n=== computing {scale} ({seconds_str}) window features ===")
        t0 = time.perf_counter()
        feats = compute_window_features(df, seconds_str, scale)
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s  ({elapsed/len(df)*1000:.3f} ms/row)")
        df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    train_df = df[df["ts"] < split_time]
    test_df = df[df["ts"] >= split_time]
    print(f"\ntrain: {len(train_df)}, test: {len(test_df)}")

    train_df.to_parquet(OUT_DIR / "train_windowed.parquet")
    test_df.to_parquet(OUT_DIR / "test_windowed.parquet")
    print(f"saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
