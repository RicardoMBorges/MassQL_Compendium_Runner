# app.py — MassQL Compendium Runner + MS/MS Viewer (Streamlit)
# -----------------------------------------------------------
# ✅ No pyarrow usage anywhere
# ✅ Upload a .mgf and multiple compendium .txt files
# ✅ Qualifier overrides (global or per-compendium name)
# ✅ Interactive MS/MS (zoom/hover via mpld3 inside Streamlit)
# ✅ Presence tables + named tables + CSV downloads (no Arrow)
# -----------------------------------------------------------

import os, re, io, math, tempfile, pathlib, json
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

from pyteomics import mgf
from massql import msql_engine, msql_fileloading

# ---------- Config: NEVER use pyarrow-backed display ----------
st.set_page_config(page_title="MassQL Compendium Viewer", layout="wide")

# ---------- Simple HTML table rendering (no pyarrow) ----------
def html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<div style='color:#666'>[empty]</div>"
    df_show = df.head(max_rows)
    return df_show.to_html(index=False, escape=False)

def download_csv_bytes(df: pd.DataFrame, sep: str = ";") -> bytes:
    if df is None or df.empty:
        return b""
    buf = io.StringIO()
    df.to_csv(buf, sep=sep, index=False)
    return buf.getvalue().encode("utf-8")

# ============================================================
# Helpers: MGF open + align
# ============================================================
def _open_mgf(path):
    try:
        return mgf.MGF(path, use_index=False)
    except TypeError:
        return mgf.MGF(path)

def _find_scan_col(df: pd.DataFrame) -> str | None:
    for cand in ("scan", "SCANS", "Scan", "Scans"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        cl = str(c).lower()
        if cl == "scan" or cl == "scans" or cl.startswith("scan"):
            return c
    return None

def align_ms2_with_mgf(ms2_df: pd.DataFrame, mgf_path: str) -> pd.DataFrame:
    rows = []
    with _open_mgf(mgf_path) as rdr:
        for mgf_seq, spec in enumerate(rdr, start=1):
            P = spec.get("params", {}) or {}
            sraw = P.get("scans")
            try:
                scan = int(str(sraw).strip()) if sraw is not None else mgf_seq
            except Exception:
                scan = mgf_seq
            pep = P.get("pepmass")
            pepmz = pep[0] if isinstance(pep, (list, tuple, np.ndarray)) else pep
            try:
                pepmz = float(pepmz)
            except Exception:
                pepmz = float("nan")
            rt_raw = P.get("rtinseconds", 0.0)
            try:
                rtsec = float(rt_raw) if rt_raw is not None else 0.0
            except Exception:
                rtsec = 0.0
            rows.append({"scan": scan, "precmz": pepmz, "rt": rtsec, "mgf_seq": mgf_seq})
    pepmass_df = pd.DataFrame(rows).astype({"scan": int})

    ms2_df = ms2_df.copy()
    scan_col = _find_scan_col(ms2_df)
    merged = None

    if scan_col is not None:
        ms2_tmp = ms2_df.copy()
        ms2_tmp["scan_new"] = pd.to_numeric(ms2_tmp[scan_col], errors="coerce")
        ms2_tmp = ms2_tmp.dropna(subset=["scan_new"])
        if not ms2_tmp.empty:
            ms2_tmp["scan_new"] = ms2_tmp["scan_new"].astype(int)
            if "scan" in ms2_tmp.columns:
                ms2_tmp = ms2_tmp.drop(columns=["scan"])
            ms2_spec = (ms2_tmp.rename(columns={"scan_new": "scan"})
                               .drop(columns=["_scan_int"], errors="ignore")
                               .drop_duplicates(subset=["scan"]))
            if "scan" in ms2_spec.columns and not ms2_spec.empty:
                merged = pepmass_df.merge(ms2_spec, on="scan", how="left", suffixes=("", "_ms2"))

    if merged is None:
        ms2_tmp = ms2_df.reset_index(drop=True).assign(_row_ix=lambda d: d.index + 1)
        ms2_spec = (ms2_tmp.groupby("_row_ix", as_index=False).first()
                            .rename(columns={"_row_ix": "mgf_seq"}))
        merged = pepmass_df.merge(ms2_spec, on="mgf_seq", how="left", suffixes=("", "_ms2"))

    for base in ("precmz", "rt"):
        ms2_col = f"{base}_ms2"
        if base not in merged.columns:
            merged[base] = merged[ms2_col] if ms2_col in merged.columns else np.nan
        elif ms2_col in merged.columns:
            merged[base] = pd.to_numeric(merged[base], errors="coerce")
            merged[ms2_col] = pd.to_numeric(merged[ms2_col], errors="coerce")
            merged[base] = merged[base].fillna(merged[ms2_col])

    merged = merged.drop(columns=["mgf_seq", "precmz_ms2", "rt_ms2",
                                  "precmz_x", "precmz_y", "rt_x", "rt_y"],
                         errors="ignore")

    merged["scan"] = pd.to_numeric(merged["scan"], errors="coerce").astype(int)
    merged["precmz"] = pd.to_numeric(merged["precmz"], errors="coerce")
    merged["rt"]     = pd.to_numeric(merged["rt"], errors="coerce")
    return merged


# ============================================================
# Load logo
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "LAABio.png"
logo_massQL = STATIC_DIR / "logo_massQL.png"

try:
    logo_massQL = Image.open(logo_massQL)
    st.sidebar.image(logo_massQL, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo_massQL not found at static/LAABio.png")


try:
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")





st.sidebar.markdown("""---""")


# ============================================================
# Parse compendiums
# ============================================================
QUERY_START_RE = re.compile(r'^\s*QUERY\b', flags=re.IGNORECASE)
_NUM = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

def parse_compendium(path: str) -> List[Dict[str, str]]:
    items = []
    current_section = None
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
            i += 1
            continue
        if QUERY_START_RE.match(stripped):
            buf = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                nxt_s = nxt.strip()
                if nxt_s.startswith("#") or QUERY_START_RE.match(nxt_s):
                    break
                if nxt_s.startswith("//"):
                    i += 1
                    continue
                buf.append(nxt)
                i += 1
            qtext = "\n".join(buf).strip()
            items.append({"section": current_section or "UnnamedSection", "query": qtext})
            continue
        i += 1
    return items

def _replace_or_append(qtext: str, key: str, value) -> str:
    pat = re.compile(rf"(?i)(:\s*{key}\s*=\s*){_NUM}")
    def _repl(m: re.Match) -> str:
        return f"{m.group(1)}{value}"
    new_q = pat.sub(_repl, qtext)
    if new_q == qtext:
        tail = f"{key}={value}"
        new_q = new_q.rstrip()
        if new_q.endswith(":"):
            return f"{new_q}{tail}"
        return f"{new_q}: {tail}"
    return new_q

def apply_qualifier_overrides(qtext: str, overrides: Dict[str, float] | None) -> str:
    if not overrides:
        return qtext
    out = qtext
    if "TOLERANCEPPM" in overrides and overrides["TOLERANCEPPM"] is not None:
        out = _replace_or_append(out, "TOLERANCEPPM", overrides["TOLERANCEPPM"])
    if "TOLERANCEMZ" in overrides and overrides["TOLERANCEMZ"] is not None:
        out = _replace_or_append(out, "TOLERANCEMZ", overrides["TOLERANCEMZ"])
    if "INTENSITYPERCENT" in overrides and overrides["INTENSITYPERCENT"] is not None:
        out = _replace_or_append(out, "INTENSITYPERCENT", overrides["INTENSITYPERCENT"])
    return out

def _select_overrides(qualifier_overrides: Dict[str, Dict[str, float]] | None, compendium_name: str) -> Dict[str, float] | None:
    if not qualifier_overrides:
        return None
    base = qualifier_overrides.get("*", {})
    spec = qualifier_overrides.get(compendium_name, {})
    return {**base, **spec} if (base or spec) else None

# ============================================================
# MassQL runner
# ============================================================
def run_compendiums(
    compendium_files: List[str],
    mgf_files: List[str],
    *,
    use_loader_frames: bool = True,
    parallel: bool = False,
    qualifier_overrides: Dict[str, Dict[str, float]] | None = None,
    source_name_map: Dict[str, str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:

    all_hits: List[pd.DataFrame] = []

    for mgf_path in mgf_files:
        # --- canonical MGF map (scan → precmz/rt) ---------------------------
        mgf_rows = []
        mgf_scans_max = 0
        with _open_mgf(mgf_path) as _rdr_mgf_:
            for _i, _spec in enumerate(_rdr_mgf_, start=1):
                _P = _spec.get("params", {}) or {}
                _s = _P.get("scans")
                try:
                    _scan = int(str(_s).strip()) if _s is not None else _i
                except Exception:
                    _scan = _i
                _pep = _P.get("pepmass")
                _pepmz = _pep[0] if isinstance(_pep, (list, tuple, np.ndarray)) else _pep
                try:
                    _pepmz = float(_pepmz)
                except Exception:
                    _pepmz = float("nan")
                _rt_raw = _P.get("rtinseconds", 0.0)
                try:
                    _rtsec = float(_rt_raw) if _rt_raw is not None else 0.0
                except Exception:
                    _rtsec = 0.0
                mgf_rows.append({"scan": _scan, "precmz_mgf": _pepmz, "rt_mgf": _rtsec})
                if _scan > mgf_scans_max:
                    mgf_scans_max = _scan
        mgf_map = pd.DataFrame(mgf_rows).astype({"scan": int})

        # --- optional loader frames + alignment -----------------------------
        ms1_df = ms2_df = None
        if use_loader_frames:
            try:
                ms1_df, ms2_df = msql_fileloading.load_data(mgf_path)
            except Exception as e:
                st.warning(f"[WARN] load_data failed for {os.path.basename(mgf_path)}: {e}")
                ms1_df, ms2_df = None, None
            if ms2_df is not None:
                ms2_df = align_ms2_with_mgf(ms2_df, mgf_path)

        # --- iterate compendiums/queries ------------------------------------
        for comp_file in compendium_files:
            comp_name = Path(comp_file).stem
            try:
                qitems = parse_compendium(comp_file)
            except Exception as e:
                st.warning(f"[WARN] Failed to parse {comp_file}: {e}")
                continue

            comp_over = _select_overrides(qualifier_overrides, comp_name)

            for q_idx, item in enumerate(qitems, start=1):
                qtext   = item["query"]
                section = item["section"] or "UnnamedSection"
                if comp_over:
                    qtext = apply_qualifier_overrides(qtext, comp_over)

                try:
                    res = msql_engine.process_query(
                        qtext,
                        mgf_path,
                        ms1_df=ms1_df,
                        ms2_df=ms2_df,
                        cache=None,
                        parallel=parallel
                    )
                except Exception as e:
                    st.info(f"[INFO] Query failed ({comp_name} :: {section} #{q_idx}) on {os.path.basename(mgf_path)}: {e}")
                    continue

                # --- normalize schema ---------------------------------------
                res = res.copy()

                # unify scan column
                if "scan" not in res.columns and "SCANS" in res.columns:
                    res["scan"] = res["SCANS"]

                # ensure key columns exist (fill missing as NA)
                for col in ("scan", "precmz", "rt", "compendium", "section", "query_idx", "source_file"):
                    if col not in res.columns:
                        res[col] = pd.NA

                # annotate meta
                res["compendium"]  = comp_name if res["compendium"].isna().all() else res["compendium"]
                res["section"]     = section   if res["section"].isna().all()     else res["section"]
                res["query_idx"]   = q_idx     if res["query_idx"].isna().all()   else res["query_idx"]

                # display name for the source file
                disp = source_name_map.get(mgf_path, os.path.basename(mgf_path)) if source_name_map else os.path.basename(mgf_path)
                res["source_file"] = disp if res["source_file"].isna().all() else res["source_file"]

                # lift uppercase metrics if present
                for c in ("precmz", "rt"):
                    if c not in res.columns and c.upper() in res.columns:
                        res[c] = res[c.upper()]

                # coerce types gently
                if "scan" in res.columns:
                    res["scan"] = pd.to_numeric(res["scan"], errors="coerce")

                for c in ("precmz", "rt"):
                    if c in res.columns:
                        res[c] = pd.to_numeric(res[c], errors="coerce")

                # backfill precmz/rt from canonical MGF
                if "scan" in res.columns:
                    res = res.merge(mgf_map, on="scan", how="left")
                    if "precmz" in res.columns:
                        res["precmz"] = res["precmz"].fillna(res["precmz_mgf"])
                    else:
                        res["precmz"] = res["precmz_mgf"]
                    if "rt" in res.columns:
                        res["rt"] = res["rt"].fillna(res["rt_mgf"])
                    else:
                        res["rt"] = res["rt_mgf"]
                    res = res.drop(columns=["precmz_mgf", "rt_mgf"])

                # optional diagnostic (no-op)
                sc_col = "scan" if "scan" in res.columns else None
                if sc_col is not None:
                    _res_sc = pd.to_numeric(res[sc_col], errors="coerce").dropna()
                    if not _res_sc.empty and _res_sc.max() < mgf_scans_max:
                        pass

                all_hits.append(res)

    # --- no hits at all ------------------------------------------------------
    if not all_hits:
        empty = pd.DataFrame(columns=["scan","precmz","rt","compendium","section","query_idx","source_file"])
        return empty, {}, empty

    # --- combine & deduplicate ----------------------------------------------
    combined = pd.concat(all_hits, ignore_index=True)

    # keep one row per (file × compendium × section × query × scan)
    dedup_keys = ["source_file","compendium","section","query_idx","scan"]
    for k in dedup_keys:
        if k not in combined.columns:
            combined[k] = pd.NA
    combined_unique = (
        combined
        .drop_duplicates(subset=dedup_keys)
        .reset_index(drop=True)
    )

    # --- presence matrices (hardened) ---------------------------------------
    need_cols = {"source_file", "scan", "compendium", "section"}
    have_cols = set(combined_unique.columns)

    presence_global = pd.DataFrame()
    presence_by_comp: Dict[str, pd.DataFrame] = {}

    if need_cols.issubset(have_cols):
        dfp = combined_unique.copy()

        # dtypes & clean
        dfp["source_file"] = dfp["source_file"].astype(str)
        dfp["scan"]        = pd.to_numeric(dfp["scan"], errors="coerce")
        dfp["compendium"]  = dfp["compendium"].astype(str)
        dfp["section"]     = dfp["section"].astype(str)

        dfp = dfp.dropna(subset=["source_file", "scan", "compendium", "section"])
        if not dfp.empty:
            dfp["hit"] = 1

            # global compendium × section matrix
            presence_global = (
                dfp.pivot_table(
                    index=["source_file","scan"],
                    columns=["compendium","section"],
                    values="hit",
                    aggfunc="max",
                    fill_value=0
                )
                .sort_index()
            )

            # per-compendium matrices
            for comp in sorted(dfp["compendium"].unique()):
                sub = dfp.loc[dfp["compendium"] == comp].copy()
                if sub.empty:
                    continue
                pres = (
                    sub.pivot_table(
                        index=["source_file","scan"],
                        columns="section",
                        values="hit",
                        aggfunc="max",
                        fill_value=0
                    )
                    .sort_index()
                )
                presence_by_comp[comp] = pres

    return combined_unique, presence_by_comp, presence_global


# ============================================================
# Pretty summaries + presence tables
# ============================================================
def summarize_results(combined_unique: pd.DataFrame) -> pd.DataFrame:
    if combined_unique.empty:
        return pd.DataFrame()
    return (combined_unique
            .groupby(["source_file","compendium","section"], as_index=False)
            .agg(n_scans=("scan","nunique"))
            .sort_values(["source_file","compendium","section"]))

def _safe_col(name: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_]+', '_', str(name)).strip('_')
    return re.sub(r'_+', '_', s) or "unnamed"

def make_named_presence_table(combined_unique: pd.DataFrame) -> pd.DataFrame:
    if combined_unique.empty:
        return pd.DataFrame(columns=["scan", "precmz", "rt"])
    base = (combined_unique.sort_values(["source_file", "scan"])
            .groupby(["source_file", "scan"], as_index=False)
            .agg(precmz=("precmz", "first"),
                 rt=("rt", "first")))
    comp_cols = []
    for comp in sorted(combined_unique["compendium"].unique()):
        col = f"compendium_{_safe_col(comp)}"
        hits = (combined_unique.loc[combined_unique["compendium"] == comp, ["source_file", "scan"]]
                .drop_duplicates().assign(**{col: comp}))
        col_df = base[["source_file", "scan"]].merge(hits, on=["source_file", "scan"], how="left")
        col_df[col] = col_df[col].fillna("")
        comp_cols.append(col_df[[col]])
    sec_cols = []
    for sec in sorted(combined_unique["section"].unique()):
        col = f"section_{_safe_col(sec)}"
        hits = (combined_unique.loc[combined_unique["section"] == sec, ["source_file", "scan"]]
                .drop_duplicates().assign(**{col: sec}))
        col_df = base[["source_file", "scan"]].merge(hits, on=["source_file", "scan"], how="left")
        col_df[col] = col_df[col].fillna("")
        sec_cols.append(col_df[[col]])
    out = pd.concat([base[["source_file","scan","precmz","rt"]]] + comp_cols + sec_cols, axis=1)
    return out.drop(columns=["source_file"])

def make_presence_table_for_compendium(combined_unique: pd.DataFrame, compendium: str) -> pd.DataFrame:
    if combined_unique.empty:
        return pd.DataFrame()
    sub = combined_unique.loc[combined_unique["compendium"] == compendium].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["hit"] = 1
    pres = sub.pivot_table(
        index=["source_file", "scan"],
        columns="section",
        values="hit",
        aggfunc="max",
        fill_value=0
    ).sort_index()
    return pres

def make_named_presence_table_for_compendium(combined_unique: pd.DataFrame, compendium: str) -> pd.DataFrame:
    sub = combined_unique.loc[combined_unique["compendium"] == compendium].copy()
    if sub.empty:
        return pd.DataFrame(columns=["scan","precmz","rt"])
    base = (sub.sort_values(["source_file", "scan"])
              .groupby(["source_file", "scan"], as_index=False)
              .agg(precmz=("precmz","first"), rt=("rt","first")))
    out_cols = [base[["source_file","scan","precmz","rt"]]]
    for sec in sorted(sub["section"].dropna().unique()):
        col = f"section_{_safe_col(sec)}"
        hits = (sub.loc[sub["section"] == sec, ["source_file","scan"]]
                  .drop_duplicates().assign(**{col: sec}))
        col_df = base[["source_file","scan"]].merge(hits, on=["source_file","scan"], how="left")
        col_df[col] = col_df[col].fillna("")
        out_cols.append(col_df[[col]])
    out = pd.concat(out_cols, axis=1).drop(columns=["source_file"])
    return out

def coverage_by_compendium(combined_unique: pd.DataFrame) -> pd.DataFrame:
    if combined_unique.empty:
        return pd.DataFrame(columns=["compendium", "n_files", "n_scans_hit", "n_hits"])
    g = combined_unique.groupby("compendium")
    return pd.DataFrame({
        "n_files": g["source_file"].nunique(),
        "n_scans_hit": g["scan"].nunique(),
        "n_hits": g.size()
    }).sort_values("n_scans_hit", ascending=False).reset_index()

# ============================================================
# MGF index + interactive MS/MS via mpld3
# ============================================================
def _build_mgf_scan_index(mgf_path: str):
    """Index MGF by scan -> {mz: np.ndarray, i: np.ndarray, params: dict} (robust to empty arrays)."""
    idx = {}
    with _open_mgf(mgf_path) as rdr:
        for seq, spec in enumerate(rdr, start=1):
            P = spec.get("params", {}) or {}

            # robust scan id
            sraw = P.get("scans")
            try:
                scan = int(str(sraw).strip()) if sraw is not None else seq
            except Exception:
                scan = seq

            # NEVER use `or` with numpy arrays; fetch explicitly
            mz = spec.get("m/z array", None)
            if mz is None:
                mz = spec.get("m/z", None)

            I = spec.get("intensity array", None)
            if I is None:
                I = spec.get("intensity", None)

            # Coerce to numpy arrays safely (empty if missing)
            if mz is None:
                mz = np.array([], dtype=float)
            else:
                mz = np.asarray(mz, dtype=float)

            if I is None:
                I = np.array([], dtype=float)
            else:
                I = np.asarray(I, dtype=float)

            idx[int(scan)] = {"mz": mz, "i": I, "params": P}
    return idx


def _plot_ms2_html_from_index(_MGF: dict, scan:int, normalize:bool=True, annotate_top_n:int=12) -> str:
    try:
        import mpld3
        from mpld3 import plugins
    except Exception as e:
        return f"<div style='color:#b00'>mpld3 not installed: {e}</div>"

    rec = _MGF.get(int(scan))
    if rec is None or rec["mz"].size == 0 or rec["i"].size == 0:
        return f"<div style='color:#b00'>No peaks found for scan {scan}</div>"

    mz, I, P = rec["mz"], rec["i"], rec["params"]
    y = I.astype(float)
    if normalize and y.size > 0 and y.max() > 0:
        y = (y / y.max()) * 100.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for x, h in zip(mz, y):
        ax.vlines(x, 0.0, h, linewidth=2.0)

    pts = ax.scatter(mz, y, s=10, alpha=0)
    labels = [f"m/z: {x:.4f}<br>I: {h:.2f}{'%' if normalize else ''}" for x, h in zip(mz, y)]
    tooltip = plugins.PointHTMLTooltip(pts, labels=labels, hoffset=10, voffset=-10)
    plugins.connect(fig, tooltip, plugins.Reset(), plugins.Zoom(), plugins.BoxZoom())

    pep = P.get("pepmass")
    pepmz = pep[0] if isinstance(pep, (list, tuple, np.ndarray)) else pep
    try:
        pepmz = float(pepmz)
    except Exception:
        pepmz = None
    if pepmz is not None and not math.isnan(pepmz):
        ax.axvline(pepmz, linestyle="--", linewidth=1.0)
        ax.text(pepmz, ax.get_ylim()[1]*0.95, f"precursor {pepmz:.4f}", rotation=90, va="top", ha="right")

    if annotate_top_n and annotate_top_n > 0:
        idx = np.argsort(y)[-annotate_top_n:]
        ymax = ax.get_ylim()[1] or 1
        for k in idx:
            ax.text(mz[k], y[k] + 0.02*ymax, f"{mz[k]:.4f}", rotation=90, va="bottom", ha="center")

    rt = P.get("rtinseconds")
    try:
        rt = float(rt) if rt is not None else None
    except Exception:
        rt = None
    title_rt = f" | RT={rt:.1f}s" if rt is not None else ""
    title_prec = f" | Precursor={pepmz:.4f}" if pepmz is not None else ""
    ax.set_title(f"Scan {scan}{title_rt}{title_prec}")
    ax.set_xlabel("m/z")
    ax.set_ylabel(f"Intensity ({'relative %' if normalize else 'a.u.'})")
    fig.tight_layout()

    import mpld3
    html = mpld3.fig_to_html(fig, no_extras=False)
    plt.close(fig)
    return html

# ============================================================
# Streamlit UI
# ============================================================
st.title("MassQL Compendium Runner + MS/MS Viewer")
st.caption("Upload an MGF and one or more MassQL compendium .txt files. Tables render via HTML (no pyarrow).")

with st.sidebar:
    st.header("1) Inputs")
    up_mgf = st.file_uploader("MGF file", type=["mgf"])
    up_comps = st.file_uploader("Compendium .txt files (multiple)", type=["txt"], accept_multiple_files=True)

    st.header("2) Qualifier overrides (applied to all)")
    colA, colB, colC = st.columns(3)
    with colA:
        tol_mz = st.number_input("TOLERANCEMZ", value=0.03, step=0.01, format="%.4f")
    with colB:
        tol_ppm = st.text_input("TOLERANCEPPM (blank = ignore)", value="")
    with colC:
        inten_pct = st.number_input("INTENSITYPERCENT", value=10, step=1)

    st.markdown("You can also add **per-compendium** overrides by name below (JSON). Example:")
    st.code('''{
  "*": {"TOLERANCEMZ": 0.03, "INTENSITYPERCENT": 10},
  "Flavonoids": {"TOLERANCEMZ": 0.02}
}''', language="json")
    per_comp_json = st.text_area("Overrides JSON (optional)", value="", height=120)

    run_btn = st.button("Run MassQL Compendiums", type="primary")

st.sidebar.markdown("""---""")

st.sidebar.markdown("""
### Citation

If you use this app or derive results from **MassQL**, please cite:

> Damiani, T., Jarmusch, A.K., Aron, A.T. *et al.*  
> **A universal language for finding mass spectrometry data patterns.**  
> *Nature Methods* **22**, 1247–1254 (2025).  
> [https://doi.org/10.1038/s41592-025-02660-z](https://doi.org/10.1038/s41592-025-02660-z)
""")


# Handle uploads to temp files
def _persist_upload(uploaded, *, keep_name: bool = True):
    """Save the uploaded file to a unique temp dir.
    Returns (path_on_disk, display_name)."""
    if not uploaded:
        return None, None
    tmpdir = tempfile.mkdtemp(prefix="massql_")
    orig = Path(uploaded.name).name if getattr(uploaded, "name", None) else "uploaded.dat"
    # sanitize: keep letters, digits, dot, dash, underscore
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', orig) or "uploaded.dat"
    outp = Path(tmpdir) / safe
    with open(outp, "wb") as f:
        f.write(uploaded.read())
    return str(outp), safe


state = st.session_state
if "combined" not in state:
    state.combined = pd.DataFrame()
    state.presence_by_comp = {}
    state.presence_global = pd.DataFrame()
    state.mgf_path = None
    state.mgf_index = None

if run_btn:
    if not up_mgf or not up_comps:
        st.error("Please upload an MGF and at least one compendium .txt.")
    else:
        # Persist uploads (NOTE: no 'suffix' arg)
        mgf_path, mgf_display = _persist_upload(up_mgf)
        comp_paths, comp_names = [], []
        for f in up_comps:
            p, n = _persist_upload(f)
            comp_paths.append(p); comp_names.append(n)

        # Build overrides
        overrides = {"*": {"TOLERANCEMZ": float(tol_mz), "INTENSITYPERCENT": int(inten_pct)}}
        if tol_ppm.strip():
            try:
                overrides["*"]["TOLERANCEPPM"] = float(tol_ppm)
            except:
                st.warning("TOLERANCEPPM not a number; ignoring.")

        if per_comp_json.strip():
            try:
                user_json = json.loads(per_comp_json)
                for k, v in user_json.items():
                    overrides[k] = {**overrides.get(k, {}), **v}
            except Exception as e:
                st.warning(f"Invalid JSON for per-compendium overrides: {e}")

        with st.spinner("Running MassQL queries..."):
            combined, presence_by_comp, presence_global = run_compendiums(
                compendium_files=comp_paths,
                mgf_files=[mgf_path],
                use_loader_frames=True,
                parallel=False,
                qualifier_overrides=overrides,
                # Build the map inline to avoid NameError
                source_name_map={mgf_path: (mgf_display or os.path.basename(mgf_path))}
            )

        state.combined = combined
        state.presence_by_comp = presence_by_comp
        state.presence_global = presence_global
        state.mgf_path = mgf_path
        state.mgf_index = _build_mgf_scan_index(mgf_path)
        st.success("Done.")


# Results area
combined = state.combined
presence_by_comp = state.presence_by_comp
presence_global = state.presence_global
mgf_index = state.mgf_index
mgf_path = state.mgf_path

if not combined.empty:
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total result rows", f"{len(combined):,}")
    with c2:
        st.metric("Unique scans (hits)", f"{combined['scan'].nunique() if 'scan' in combined.columns else 0:,}")
    with c3:
        st.metric("Compendiums", f"{combined['compendium'].nunique():,}")
    with c4:
        st.metric("Sections", f"{combined['section'].nunique():,}")

    st.markdown("**Per file × compendium × section**")
    summary = summarize_results(combined)
    st.markdown(html_table(summary, 200), unsafe_allow_html=True)
    st.download_button("Download summary.csv", data=download_csv_bytes(summary), file_name="summary.csv")

    st.markdown("---")
    st.markdown("### Global presence (compendium × section)")
    if not presence_global.empty:
        # Flatten MultiIndex columns to strings
        pg = presence_global.copy()
        pg.columns = [f"{c[0]} :: {c[1]}" for c in pg.columns.to_list()]
        pg = pg.reset_index()
        st.markdown(html_table(pg, 200), unsafe_allow_html=True)
        st.download_button("Download presence_global.csv",
                           data=download_csv_bytes(pg), file_name="presence_global.csv")
    else:
        st.info("No global presence matrix.")

    st.markdown("---")
    st.markdown("### Presence by compendium")
    for comp, pres in presence_by_comp.items():
        st.markdown(f"**{comp}**")
        pres2 = pres.reset_index()
        st.markdown(html_table(pres2, 150), unsafe_allow_html=True)
        st.download_button(f"Download presence_{_safe_col(comp)}.csv",
                           data=download_csv_bytes(pres2),
                           file_name=f"presence_{_safe_col(comp)}.csv")

    st.markdown("---")
    st.markdown("### Named presence table (compendium and section labels in cells)")
    named = make_named_presence_table(combined)
    st.markdown(html_table(named, 200), unsafe_allow_html=True)
    st.download_button("Download combined_wide.csv",
                       data=download_csv_bytes(named), file_name="combined_wide.csv")

    st.markdown("---")
    st.markdown("### Coverage by compendium")
    cov = coverage_by_compendium(combined)
    st.markdown(html_table(cov, 200), unsafe_allow_html=True)
    st.download_button("Download coverage.csv", data=download_csv_bytes(cov), file_name="coverage.csv")

    # ===================== MS/MS viewer =====================
    st.markdown("---")
    st.subheader("Interactive MS/MS viewer (mpld3)")

    # Scan options: prefer scans that had hits; fallback to all indexed scans
    if "scan" in combined.columns:
        scans = sorted(map(int, pd.to_numeric(combined["scan"], errors="coerce").dropna().unique()))
    else:
        scans = []
    if not scans and mgf_index:
        scans = sorted(mgf_index.keys())

    if scans:
        colL, colR = st.columns([2, 1])
        with colL:
            chosen_scan = st.selectbox("Scan", scans, index=0)
        with colR:
            normalize = st.checkbox("Normalize to 100%", value=True)
            topn = st.slider("Annotate top-N peaks", min_value=0, max_value=40, value=12, step=1)

        if mgf_index:
            html = _plot_ms2_html_from_index(mgf_index, int(chosen_scan), normalize=normalize, annotate_top_n=topn)
            components.html(html, height=420, scrolling=True)
        else:
            st.warning("MGF index not available to render spectrum.")

        # Per-scan MassQL hits table
        st.markdown("**MassQL hits for selected scan**")
        dfscan = combined.copy()
        if "scan" in dfscan.columns:
            dfscan = dfscan[pd.to_numeric(dfscan["scan"], errors="coerce") == int(chosen_scan)]
            keep = [c for c in ("source_file","scan","precmz","rt","compendium","section","query_idx") if c in dfscan.columns]
            if keep:
                dfscan = dfscan[keep].drop_duplicates().sort_values(["source_file","compendium","section","query_idx"])
        st.markdown(html_table(dfscan, 300), unsafe_allow_html=True)
        st.download_button("Download scan_hits.csv",
                           data=download_csv_bytes(dfscan), file_name=f"scan_{chosen_scan}_hits.csv")
    else:
        st.info("No scans available to display.")

else:
    st.info("Upload inputs in the sidebar and press **Run MassQL Compendiums**.")
