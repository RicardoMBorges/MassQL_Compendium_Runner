# app.py — MassQL Compendium Runner + MS/MS Viewer (Streamlit)
# -----------------------------------------------------------
# ✅ No pyarrow usage anywhere
# ✅ Upload a .mgf (.mzXML or .mzML) and multiple compendium .txt files
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

from pyteomics import mgf, mzml, mzxml
import pymzml
from massql import msql_engine, msql_fileloading

# --- Patch MassQL MGF loader to force correct scan numbers ---
from matchms.importing import load_from_mgf

def patched_load_from_mgf(path):
    spectra = list(load_from_mgf(path))
    for i, spec in enumerate(spectra, start=1):
        if not hasattr(spec, "metadata"):
            continue
        # Force proper scan numbering
        spec.metadata["scans"] = i  
    return spectra

# Override MassQL internal loader
msql_fileloading.load_from_mgf = patched_load_from_mgf


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
# Helpers: MGF (MGF / mzML / mzXML) open + align
# ============================================================
def _open_mgf(path):
    """Version-safe MGF opener (handles older Pyteomics without use_index)."""
    try:
        # Newer pyteomics: supports use_index
        return mgf.MGF(path, use_index=False)
    except TypeError:
        # Older pyteomics: no use_index argument
        return mgf.MGF(path)

def _open_ms(path: str):
    """
    Open MS file (MGF, mzML, mzXML) with pyteomics.
    """
    ext = Path(path).suffix.lower()
    if ext == ".mgf":
        try:
            return mgf.MGF(path, use_index=False)
        except TypeError:
            return mgf.MGF(path)
    elif ext == ".mzml":
        return mzml.MzML(path)
    elif ext == ".mzxml":
        return mzxml.MzXML(path)
    else:
        # fallback: assume MGF
        try:
            return mgf.MGF(path, use_index=False)
        except TypeError:
            return mgf.MGF(path)

def _extract_meta_from_spec(spec: dict, seq: int, ext: str):
    """
    Return (scan, precursor_mz, rt_seconds) from a spectrum dict
    for MGF / mzML / mzXML in a best-effort way.
    """
    # ---------- MGF ----------
    if ext == ".mgf":
        P = spec.get("params", {}) or {}

        # scan
        sraw = P.get("scans")
        try:
            scan = int(str(sraw).strip()) if sraw is not None else seq
        except Exception:
            scan = seq

        # ---------- FIXED PEPMASS HANDLING ----------
        pep = P.get("pepmass")

        # pep can be: number, list, tuple, array, or "mz intensity"
        if isinstance(pep, (list, tuple, np.ndarray)):
            raw = pep[0] if len(pep) > 0 else None
        else:
            raw = pep

        mz_val = None
        if raw is not None:
            # Take only the first token (handles "221.025 1000.0")
            tok = str(raw).strip().split()[0]
            try:
                mz_val = float(tok)
            except Exception:
                mz_val = None

        pepmz = mz_val if mz_val is not None else float("nan")
        # --------------------------------------------

        # RT (in seconds if available, else 0)
        rt_raw = P.get("rtinseconds", P.get("rt", None))
        try:
            rtsec = float(rt_raw) if rt_raw is not None else 0.0
        except Exception:
            rtsec = 0.0

        return scan, pepmz, rtsec

    # ---------- mzML / mzXML ----------
    # scan: prefer 'index' (0-based), else fall back to enumeration
    scan = spec.get("index")
    try:
        scan = int(scan) + 1 if scan is not None else seq
    except Exception:
        scan = seq

    pepmz = float("nan")
    rtsec = 0.0

    # precursor m/z (very rough but robust for typical mzML/mzXML)
    try:
        pl = spec.get("precursorList") or spec.get("precursors")
        precs = None
        if isinstance(pl, dict):
            precs = pl.get("precursor")
        elif isinstance(pl, list):
            precs = pl
        if isinstance(precs, list) and precs:
            p0 = precs[0]
            sil = p0.get("selectedIonList") or {}
            ions = sil.get("selectedIon") or []
            if isinstance(ions, list) and ions:
                ion0 = ions[0]
                for key in ("selected ion m/z", "selected_ion_mz", "m/z"):
                    if key in ion0:
                        pepmz = float(ion0[key])
                        break
    except Exception:
        pass

    # RT from scanList / cvParam if present (otherwise stays 0.0)
    try:
        scan_list = spec.get("scanList", {})
        scans = scan_list.get("scan")
        if isinstance(scans, list) and scans:
            s0 = scans[0]
            # direct field
            if "scan start time" in s0:
                rt_val = s0["scan start time"]
                rtsec = float(rt_val)
            # cvParam style
            elif "cvParam" in s0:
                for cv in s0["cvParam"]:
                    name = str(cv.get("name", "")).lower()
                    if "scan start time" in name:
                        val = cv.get("value")
                        unit = str(cv.get("unitName", "")).lower()
                        if val is not None:
                            rtsec = float(val)
                            if "min" in unit:
                                rtsec *= 60.0
                        break
    except Exception:
        pass

    return scan, pepmz, rtsec


def _find_scan_col(df: pd.DataFrame) -> str | None:
    for cand in ("scan", "SCANS", "Scan", "Scans"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        cl = str(c).lower()
        if cl == "scan" or cl == "scans" or cl.startswith("scan"):
            return c
    return None



def _count_ms_spectra(ms_path: str) -> int:
    try:
        n = 0
        with _open_ms(ms_path) as rdr:
            for _ in rdr:
                n += 1
        return n
    except Exception as e:
        st.error(f"[MS] Could not read file: {e}")
        return -1


def diagnostics_check(ms_path: str, comp_paths: List[str]) -> pd.DataFrame:
    """Return a small table with: compendium file, bytes, parsed QUERY count, first section names."""
    rows = []

    # MS basics
    if ms_path and os.path.exists(ms_path):
        st.info(f"[Diagnostics] MS file: **{os.path.basename(ms_path)}**  |  size: {os.path.getsize(ms_path):,} bytes")
        n_spec = _count_ms_spectra(ms_path)
        if n_spec >= 0:
            st.info(f"[Diagnostics] Spectra indexed: **{n_spec}**")

    # Compendiums: parse count
    for p in comp_paths:
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="ignore")
            qitems = parse_compendium(p)
            sects = sorted({(qi.get("section") or "UnnamedSection") for qi in qitems})[:5]
            rows.append({
                "file": os.path.basename(p),
                "bytes": len(txt.encode("utf-8")),
                "queries_parsed": len(qitems),
                "sections_preview": ", ".join(sects)
            })
        except Exception as e:
            rows.append({
                "file": os.path.basename(p),
                "bytes": None,
                "queries_parsed": 0,
                "sections_preview": f"[parse error: {e}]"
            })

    return pd.DataFrame(rows)


def smoke_test_massql(ms_path: str) -> tuple[bool, str]:
    if not ms_path or not os.path.exists(ms_path):
        return False, "MS path missing"
    try:
        q = "QUERY scaninfo(MS2DATA)"
        res = msql_engine.process_query(q, ms_path, ms1_df=None, ms2_df=None, cache=None, parallel=True)
        return (not res.empty), f"Rows: {len(res)}"
    except Exception as e:
        return False, f"MassQL error: {e}"

def align_ms2_with_mgf(ms2_df: pd.DataFrame, ms_path: str) -> pd.DataFrame:
    """
    Align ms2_df with canonical data extracted from an MS file (MGF, mzML, mzXML).
    For MGF, also attaches NAME and SPECTRUMID if present.
    """

    def _get_ci(d: dict, key: str):
        """Case-insensitive key getter."""
        if not d:
            return None
        key = key.lower()
        for k, v in d.items():
            try:
                if str(k).lower() == key:
                    return v
            except Exception:
                pass
        return None

    ext = Path(ms_path).suffix.lower()

    # --- Build reference table from raw MS file ----------------------------
    rows = []
    with _open_ms(ms_path) as rdr:
        for seq, spec in enumerate(rdr, start=1):
            scan, pepmz, rtsec = _extract_meta_from_spec(spec, seq, ext)

            nm = None
            sid = None
            if ext == ".mgf":
                P = spec.get("params", {}) or {}
                nm  = _get_ci(P, "NAME")
                sid = _get_ci(P, "SPECTRUMID")

            rows.append({
                "scan": scan,
                "precmz": pepmz,
                "rt": rtsec,
                "seq": seq,
                "NAME": nm,
                "SPECTRUMID": sid
            })

    ref = pd.DataFrame(rows)
    if ref.empty:
        return ms2_df

    ref["scan"] = pd.to_numeric(ref["scan"], errors="coerce").astype("Int64")

    # --- Align ms2_df ------------------------------------------------------
    ms2_df = ms2_df.copy()
    scan_col = _find_scan_col(ms2_df)
    if scan_col is not None:
        ms2_tmp = ms2_df.copy()
        ms2_tmp["scan_new"] = pd.to_numeric(ms2_tmp[scan_col], errors="coerce").astype("Int64")
        ms2_tmp = ms2_tmp.dropna(subset=["scan_new"])
        if "scan" in ms2_tmp.columns:
            ms2_tmp = ms2_tmp.drop(columns=["scan"])
        ms2_spec = (
            ms2_tmp.rename(columns={"scan_new": "scan"})
                   .drop(columns=["_scan_int"], errors="ignore")
                   .drop_duplicates(subset=["scan"])
        )
        merged = ref.merge(ms2_spec, on="scan", how="left", suffixes=("", "_ms2"))
    else:
        # fallback: align by file order (seq)
        ms2_tmp = ms2_df.reset_index(drop=True).assign(_row_ix=lambda d: d.index + 1)
        ms2_spec = (
            ms2_tmp.groupby("_row_ix", as_index=False).first()
                   .rename(columns={"_row_ix": "seq"})
        )
        merged = ref.merge(ms2_spec, on="seq", how="left", suffixes=("", "_ms2"))

    # Backfill numeric columns from potential *_ms2 sources if they exist
    for base in ("precmz", "rt"):
        ms2_col = f"{base}_ms2"
        if base not in merged.columns:
            merged[base] = merged[ms2_col] if ms2_col in merged.columns else np.nan
        elif ms2_col in merged.columns:
            merged[base] = pd.to_numeric(merged[base], errors="coerce")
            merged[ms2_col] = pd.to_numeric(merged[ms2_col], errors="coerce")
            merged[base] = merged[base].fillna(merged[ms2_col])

    merged = merged.drop(columns=["seq", "precmz_ms2", "rt_ms2",
                                  "precmz_x", "precmz_y", "rt_x", "rt_y"],
                         errors="ignore")

    merged["scan"]   = pd.to_numeric(merged["scan"], errors="coerce").astype("Int64")
    merged["precmz"] = pd.to_numeric(merged["precmz"], errors="coerce")
    merged["rt"]     = pd.to_numeric(merged["rt"], errors="coerce")

    return merged

# ============================================================
# Helper: convert MS1-only MGF → mzML for MassQL MS1 queries
# ============================================================
# ============================================================
# Helper: convert MS1-only MGF → mzML for MassQL MS1 queries
# ============================================================
def convert_ms1_mgf_to_mzml(mgf_path: str) -> str:
    """
    Convert an MS1-only MGF file to a simple mzML with one MS1 spectrum per
    MGF entry, so MassQL can query MS1DATA / MS1MZ.

    Returns the path to the created .mzML file.
    If the MGF has no spectra, returns the original mgf_path.
    """
    # Read spectra using the same helper used everywhere else
    spectra = []
    try:
        with _open_mgf(mgf_path) as rdr:
            for spec in rdr:
                spectra.append(spec)
    except Exception:
        # If we can't even open/parse, bail out and keep the original
        return mgf_path

    if not spectra:
        # Nothing to convert; just return the original path.
        return mgf_path

    base = os.path.splitext(mgf_path)[0]
    mzml_path = base + "__converted_ms1.mzML"

    with open(mzml_path, "wb") as fh:
        writer = pymzml.MzMLWriter(fh, mzml_mode="indexed")
        with writer:
            writer.controlled_vocabularies()
            writer.write_run_header()
            writer.write_spectrum_list_header(count=len(spectra))

            for idx, spec in enumerate(spectra, start=1):
                mz_arr = np.asarray(spec.get("m/z array", []), dtype=float)
                i_arr  = np.asarray(spec.get("intensity array", []), dtype=float)

                params = spec.get("params", {}) or {}
                rt_raw = params.get("rtinseconds", 0.0)
                try:
                    rt = float(rt_raw)
                except Exception:
                    rt = 0.0

                writer.write_spectrum(
                    mz_arr,
                    i_arr,
                    id=f"scan={idx}",
                    params={
                        "ms level": 1,
                        "scan start time": (rt, "second"),
                        "centroid spectrum": None,
                        "total ion current": float(i_arr.sum()),
                    },
                )

            writer.write_spectrum_list_footer()
            writer.write_run_footer()

    return mzml_path


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
    mgf_files: List[str],   # continua esse nome para não quebrar chamadas
    *,
    use_loader_frames: bool = True,
    parallel: bool = False,
    qualifier_overrides: Dict[str, Dict[str, float]] | None = None,
    source_name_map: Dict[str, str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:

    def _get_ci(d: dict, key: str):
        if not d:
            return None
        k = key.lower()
        for kk, vv in d.items():
            try:
                if str(kk).lower() == k:
                    return vv
            except Exception:
                pass
        return None

    all_hits: List[pd.DataFrame] = []

    for ms_path in mgf_files:
        ms_path = os.fspath(ms_path)
        ext = Path(ms_path).suffix.lower()

        # --- canonical map (scan → precmz / rt / NAME / SPECTRUMID) -------
        ms_rows = []
        ms_scans_max = 0
        with _open_ms(ms_path) as _rdr_ms_:
            for _i, _spec in enumerate(_rdr_ms_, start=1):
                _scan, _pepmz, _rtsec = _extract_meta_from_spec(_spec, _i, ext)

                _nm = None
                _sid = None
                if ext == ".mgf":
                    P = _spec.get("params", {}) or {}
                    _nm  = _get_ci(P, "NAME")
                    _sid = _get_ci(P, "SPECTRUMID")

                ms_rows.append({
                    "scan": _scan,
                    "precmz_mgf": _pepmz,
                    "rt_mgf": _rtsec,
                    "NAME": _nm,
                    "SPECTRUMID": _sid,
                })
                try:
                    _scan_int = int(_scan)
                except Exception:
                    _scan_int = _i
                if _scan_int > ms_scans_max:
                    ms_scans_max = _scan_int

        ms_map = pd.DataFrame(ms_rows)
        if not ms_map.empty:
            ms_map["scan"] = pd.to_numeric(ms_map["scan"], errors="coerce").astype("Int64")

        # --- optional loader frames + alignment -----------------------------
        ms1_df = ms2_df = None
        if use_loader_frames:
            try:
                ms1_df, ms2_df = msql_fileloading.load_data(ms_path)
            except Exception as e:
                st.warning(f"[WARN] load_data failed for {os.path.basename(ms_path)}: {e}")
                ms1_df, ms2_df = None, None
            if ms2_df is not None:
                ms2_df = align_ms2_with_mgf(ms2_df, ms_path)

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
                        ms_path,
                        ms1_df=ms1_df,
                        ms2_df=ms2_df,
                        cache=True,
                        parallel=parallel
                    )
                except Exception as e:
                    st.info(f"[INFO] Query failed ({comp_name} :: {section} #{q_idx}) on {os.path.basename(ms_path)}: {e}")
                    continue

                if res is None or len(res) == 0:
                    continue

                # --- normalize schema ---------------------------------------
                res = res.copy()

                # unify scan column
                scan_variants = [c for c in res.columns if str(c).lower() in ("scan", "scans")]
                if scan_variants:
                    src = scan_variants[0]
                    res["scan"] = res[src]
                elif "spectrumindex" in (c.lower() for c in res.columns):
                    true_col = [c for c in res.columns if c.lower() == "spectrumindex"][0]
                    res["scan"] = pd.to_numeric(res[true_col], errors="coerce").add(1)
                elif "spectrum_id" in (c.lower() for c in res.columns):
                    true_col = [c for c in res.columns if c.lower() == "spectrum_id"][0]
                    res["scan"] = pd.to_numeric(res[true_col], errors="coerce")

                for col in ("precmz", "rt", "compendium", "section", "query_idx", "source_file"):
                    if col not in res.columns:
                        res[col] = pd.NA

                res["compendium"]  = comp_name if res["compendium"].isna().all() else res["compendium"]
                res["section"]     = section   if res["section"].isna().all()     else res["section"]
                res["query_idx"]   = q_idx     if res["query_idx"].isna().all()   else res["query_idx"]

                disp = source_name_map.get(ms_path, os.path.basename(ms_path)) if source_name_map else os.path.basename(ms_path)
                res["source_file"] = disp if res["source_file"].isna().all() else res["source_file"]

                # uppercase fallbacks
                for c in ("precmz", "rt"):
                    if c not in res.columns and c.upper() in res.columns:
                        res[c] = res[c.upper()]

                if "scan" in res.columns:
                    res["scan"] = pd.to_numeric(res["scan"], errors="coerce")

                for c in ("precmz", "rt"):
                    if c in res.columns:
                        res[c] = pd.to_numeric(res[c], errors="coerce")

                # --- backfill from canonical map ----------------------------
                if "scan" in res.columns and not ms_map.empty:
                    res = res.merge(ms_map, on="scan", how="left")

                    if "precmz" in res.columns:
                        res["precmz"] = res["precmz"].fillna(res["precmz_mgf"])
                    else:
                        res["precmz"] = res["precmz_mgf"]

                    if "rt" in res.columns:
                        res["rt"] = res["rt"].fillna(res["rt_mgf"])
                    else:
                        res["rt"] = res["rt_mgf"]

                    res = res.drop(columns=["precmz_mgf", "rt_mgf"], errors="ignore")

                # optional diagnostic
                sc_col = "scan" if "scan" in res.columns else None
                if sc_col is not None:
                    _res_sc = pd.to_numeric(res[sc_col], errors="coerce").dropna()
                    if not _res_sc.empty and ms_scans_max > 0 and _res_sc.max() < ms_scans_max:
                        pass

                all_hits.append(res)

    if not all_hits:
        empty = pd.DataFrame(columns=[
            "scan","precmz","rt","compendium","section","query_idx","source_file","NAME","SPECTRUMID"
        ])
        return empty, {}, empty

    combined = pd.concat(all_hits, ignore_index=True)

    dedup_keys = ["source_file","compendium","section","query_idx","scan"]
    for k in dedup_keys:
        if k not in combined.columns:
            combined[k] = pd.NA
    combined_unique = (
        combined
        .drop_duplicates(subset=dedup_keys)
        .reset_index(drop=True)
    )

    presence_global = pd.DataFrame()
    presence_by_comp: Dict[str, pd.DataFrame] = {}

    need_cols = {"source_file", "scan", "compendium", "section"}
    have_cols = set(combined_unique.columns)

    if need_cols.issubset(have_cols):
        dfp = combined_unique.copy()
        dfp["source_file"] = dfp["source_file"].astype(str)
        dfp["scan"]        = pd.to_numeric(dfp["scan"], errors="coerce")
        dfp["compendium"]  = dfp["compendium"].astype(str)
        dfp["section"]     = dfp["section"].astype(str)
        dfp = dfp.dropna(subset=["source_file","scan","compendium","section"])

        if not dfp.empty:
            dfp["hit"] = 1

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
def _build_ms_scan_index(ms_path: str):
    """Index MS file by scan -> {mz: np.ndarray, i: np.ndarray, params: dict} for MGF/mzML/mzXML."""
    idx = {}
    ext = Path(ms_path).suffix.lower()

    with _open_ms(ms_path) as rdr:
        for seq, spec in enumerate(rdr, start=1):
            scan, pepmz, rtsec = _extract_meta_from_spec(spec, seq, ext)

            if ext == ".mgf":
                P = spec.get("params", {}) or {}
                params = {
                    "pepmass": P.get("pepmass"),
                    "rtinseconds": P.get("rtinseconds"),
                    "scans": P.get("scans"),
                }

                # Avoid boolean evaluation of NumPy arrays
                mz = spec.get("m/z array", None)
                if mz is None:
                    mz = spec.get("m/z", None)
                if mz is None:
                    mz = []

                I = spec.get("intensity array", None)
                if I is None:
                    I = spec.get("intensity", None)
                if I is None:
                    I = []

            else:
                P = {}
                params = {
                    "pepmass": [pepmz] if pepmz is not None and not (isinstance(pepmz,float) and math.isnan(pepmz)) else None,
                    "rtinseconds": rtsec,
                    "id": spec.get("id"),
                }
                mz = spec.get("m/z array", [])
                I  = spec.get("intensity array", [])

            mz = np.asarray(mz, dtype=float) if mz is not None else np.array([], dtype=float)
            I  = np.asarray(I,  dtype=float) if I  is not None else np.array([], dtype=float)

            try:
                scan_int = int(scan)
            except Exception:
                scan_int = seq

            idx[scan_int] = {"mz": mz, "i": I, "params": params}

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
st.caption("Upload an MS file (.mgf, .mzML, .mzXML) and one or more MassQL compendium .txt files. Tables render via HTML (no pyarrow).")

with st.sidebar:
    st.header("1) Inputs")
    up_ms_files = st.file_uploader(
        "MS files (.mgf / .mzML / .mzXML)",
        help=".mgf files exported using 'Export Scans' and selecting MS1 data can be used.",
        type=["mgf", "mzml", "mzxml"],
        accept_multiple_files=True
    )
    # NEW: option to convert MS1-only MGF to mzML
    convert_ms1_mgf_flag = st.checkbox(
        "Convert MS1-only MGF to mzML for MS1 queries",
        value=False,
        help="If checked, any uploaded .mgf file will be converted to a simple MS1 mzML before running MassQL."
    )

    up_comps = st.file_uploader(
        "Compendium .txt files (multiple)",
        type=["txt"],
        accept_multiple_files=True,
        help=".txt file with separate MassQL queries.",
    )

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
    state.ms_paths = []          # list of all MS file paths on disk
    state.ms_indexes = {}        # dict[path] -> scan index from _build_ms_scan_index
    state.source_name_map = {}   # dict[path] -> display name
    state.last_comp_paths = []   # compendium file paths


if run_btn:
    if not up_ms_files or not up_comps:
        st.error("Please upload at least one MS file and one compendium .txt.")
    else:
        # Persist MS uploads
        ms_paths, ms_displays = [], []
        for f in up_ms_files:
            p, n = _persist_upload(f)
            ms_paths.append(p)
            ms_displays.append(n)

        # Persist compendiums
        comp_paths, comp_names = [], []
        for f in up_comps:
            p, n = _persist_upload(f)
            comp_paths.append(p)
            comp_names.append(n)
            
        # Optionally convert MS1-only MGF → mzML for MassQL MS1 queries
        converted_files = []  # list of (display_name, converted_path)
        ms_paths_effective = []

        for p, disp in zip(ms_paths, ms_displays):
            ext = Path(p).suffix.lower()
            if convert_ms1_mgf_flag and ext == ".mgf":
                try:
                    mzml_p = convert_ms1_mgf_to_mzml(p)

                    # Only treat as "converted" if we actually got a new mzML file
                    if Path(mzml_p).suffix.lower() == ".mzml" and mzml_p != p:
                        ms_paths_effective.append(mzml_p)
                        converted_files.append((disp, mzml_p))
                        st.info(
                            f"Converted MS1 MGF → mzML for {disp} → {os.path.basename(mzml_p)}"
                        )
                    else:
                        ms_paths_effective.append(p)
                        st.warning(
                            f"MGF→mzML conversion skipped for {disp}: "
                            "no spectra found or conversion did not produce a new mzML file."
                        )
                except Exception as e:
                    ms_paths_effective.append(p)
                    st.warning(
                        f"MGF→mzML conversion failed for {disp}; using original MGF. "
                        f"Error: {e}"
                    )
            else:
                ms_paths_effective.append(p)



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

        ## Map: full path -> human display name (using effective paths!)
        source_name_map = {p: (d or os.path.basename(p)) for p, d in zip(ms_paths_effective, ms_displays)}

        with st.spinner("Running MassQL queries..."):
            combined, presence_by_comp, presence_global = run_compendiums(
                compendium_files=comp_paths,
                mgf_files=ms_paths_effective,  # <<< use effective paths here
                use_loader_frames=True,
                parallel=False,
                qualifier_overrides=overrides,
                source_name_map=source_name_map
            )

        # Build one scan index per MS file for the viewer
        ms_indexes = {p: _build_ms_scan_index(p) for p in ms_paths_effective}

        # Save in state
        state.combined = combined
        state.presence_by_comp = presence_by_comp
        state.presence_global = presence_global
        state.ms_paths = ms_paths_effective
        state.ms_indexes = ms_indexes
        state.source_name_map = source_name_map
        state.last_comp_paths = comp_paths

        st.success("Done.")

        # Let the user download any converted mzML files
        if converted_files:
            st.markdown("### Download converted mzML files")
            for disp, mzml_p in converted_files:
                label = f"Download {Path(disp).stem}__converted_ms1.mzML"
                with open(mzml_p, "rb") as fh:
                    st.download_button(
                        label=label,
                        data=fh.read(),
                        file_name=os.path.basename(mzml_p),
                        mime="application/octet-stream",
                        key=f"dl_{os.path.basename(mzml_p)}",
                    )

        # ---------- DIAGNOSTICS if nothing came back ----------
        if state.combined is None or state.combined.empty:
            with st.expander("Diagnostics (why are there no results?)", expanded=True):
                st.warning("No MassQL hits were returned. Here are some checks:")
                # just run diagnostics on the first MS file for now
                first_ms = state.ms_paths[0] if state.ms_paths else None
                diag = diagnostics_check(first_ms, state.last_comp_paths) if first_ms else pd.DataFrame()
                if not diag.empty:
                    st.markdown(html_table(diag, 200), unsafe_allow_html=True)

                if first_ms:
                    ok, msg = smoke_test_massql(first_ms)
                    if ok:
                        st.success(f"MassQL smoke test: OK — {msg}")
                    else:
                        st.error(f"MassQL smoke test failed — {msg}")
                st.caption("Tip: verify your compendium files contain lines starting with `QUERY`, and that tolerances aren’t too strict.")


# ============================================================
# Results area
# ============================================================
combined = state.combined
presence_by_comp = state.presence_by_comp
presence_global = state.presence_global
ms_indexes = getattr(state, "ms_indexes", {})
source_name_map = getattr(state, "source_name_map", {})

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

    # 1) Choose which MS file (by display name = source_file)
    # combined["source_file"] already stores display names from source_name_map
    sources = sorted(combined["source_file"].astype(str).unique())
    if not sources:
        st.info("No sources available for MS/MS viewer.")
    else:
        col_src, col_opts = st.columns([2, 3])
        with col_src:
            chosen_source = st.selectbox("Source file", sources, index=0)

        # map display name -> real path
        name_to_path = {v: k for k, v in source_name_map.items()}
        ms_path = name_to_path.get(chosen_source)
        mgf_index = ms_indexes.get(ms_path, {}) if ms_path else {}

        # 2) Scans: prefer those with MassQL hits for this source
        df_src = combined[combined["source_file"] == chosen_source].copy()
        if "scan" in df_src.columns:
            scans = sorted(
                map(int, pd.to_numeric(df_src["scan"], errors="coerce").dropna().unique())
            )
        else:
            scans = []

        # Fallback: if no hit scans, use all scans from that file index
        if not scans and mgf_index:
            scans = sorted(mgf_index.keys())

        if scans:
            with col_opts:
                chosen_scan = st.selectbox("Scan", scans, index=0)
                normalize = st.checkbox("Normalize to 100%", value=True)
                topn = st.slider("Annotate top-N peaks", min_value=0, max_value=40, value=12, step=1)

            if mgf_index:
                html = _plot_ms2_html_from_index(mgf_index, int(chosen_scan),
                                                 normalize=normalize, annotate_top_n=topn)
                components.html(html, height=420, scrolling=True)
            else:
                st.warning("MS index not available to render spectrum for this file.")

            # Per-scan MassQL hits table, filtered by BOTH source_file and scan
            st.markdown("**MassQL hits for selected scan**")
            dfscan = df_src.copy()
            if "scan" in dfscan.columns:
                dfscan = dfscan[pd.to_numeric(dfscan["scan"], errors="coerce") == int(chosen_scan)]
                keep_base = ["source_file","scan","precmz","rt","compendium","section","query_idx"]
                keep_extra = [c for c in ("NAME","SPECTRUMID") if c in dfscan.columns]
                keep = [c for c in keep_base + keep_extra if c in dfscan.columns]
                if keep:
                    dfscan = (dfscan[keep]
                              .drop_duplicates()
                              .sort_values(["source_file","compendium","section","query_idx"]))
            st.markdown(html_table(dfscan, 300), unsafe_allow_html=True)
            st.download_button(
                "Download scan_hits.csv",
                data=download_csv_bytes(dfscan),
                file_name=f"{_safe_col(chosen_source)}_scan_{chosen_scan}_hits.csv"
            )
        else:
            st.info("No scans available to display for this file.")
else:
    st.info("Upload inputs in the sidebar and press **Run MassQL Compendiums**.")







