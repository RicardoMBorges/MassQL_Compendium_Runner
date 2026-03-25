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

from massql import msql_fileloading
from massql import msql_engine   
#from matchms.importing import load_from_mgf

def patched_load_from_mgf(path):
    """
    Lightweight MGF loader using pyteomics only.
    Returns a list of spectrum-like dicts compatible with our workflow.
    """
    spectra = []

    with _open_mgf(path) as reader:
        for i, spec in enumerate(reader, start=1):
            params = spec.get("params", {}) or {}

            # force scan numbering
            params["scans"] = i

            # default charge
            if "charge" not in params or params["charge"] in (None, "", "0"):
                params["charge"] = 1

            # normalize precursor m/z
            pep = params.get("pepmass")
            precursor_mz = None

            if isinstance(pep, (list, tuple, np.ndarray)):
                if len(pep) > 0:
                    try:
                        precursor_mz = float(str(pep[0]).split()[0])
                    except Exception:
                        precursor_mz = None
            elif isinstance(pep, str):
                try:
                    precursor_mz = float(pep.split()[0])
                except Exception:
                    precursor_mz = None
            elif pep is not None:
                try:
                    precursor_mz = float(pep)
                except Exception:
                    precursor_mz = None

            if precursor_mz is not None:
                params["precursor_mz"] = precursor_mz

            spectra.append(
                {
                    "m/z array": np.asarray(spec.get("m/z array", []), dtype=float),
                    "intensity array": np.asarray(spec.get("intensity array", []), dtype=float),
                    "params": params,
                }
            )

    return spectra

# override in MassQL
msql_fileloading.load_from_mgf = patched_load_from_mgf

# ---------- Config: NEVER use pyarrow-backed display ----------
st.set_page_config(page_title="MassQL Compendium Viewer", layout="wide")

# ---------- Simple HTML table rendering (no pyarrow) ----------
def html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
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

        # IMPORTANT: force sequential scan numbers (1..N) to match MassQL
        # and the patched load_from_mgf loader.
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

    if ms_path and os.path.exists(ms_path):
        st.info(f"[Diagnostics] MS file: **{os.path.basename(ms_path)}**  |  size: {os.path.getsize(ms_path):,} bytes")
        n_spec = _count_ms_spectra(ms_path)
        if n_spec >= 0:
            st.info(f"[Diagnostics] Spectra indexed: **{n_spec}**")

    for p in comp_paths:
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="ignore")
            qitems, warns = parse_compendium_auto(txt, fallback_name=Path(p).stem)
            sects = sorted({(qi.get("section") or "UnnamedSection") for qi in qitems})[:5]
            rows.append({
                "file": os.path.basename(p),
                "bytes": len(txt.encode("utf-8")),
                "queries_parsed": len(qitems),
                "sections_preview": ", ".join(sects),
                "warnings_preview": " | ".join(warns[:3]) if warns else ""
            })
        except Exception as e:
            rows.append({
                "file": os.path.basename(p),
                "bytes": None,
                "queries_parsed": 0,
                "sections_preview": f"[parse error: {e}]",
                "warnings_preview": ""
            })

    return pd.DataFrame(rows)


def smoke_test_massql(ms_path: str) -> tuple[bool, str]:
    if not ms_path or not os.path.exists(ms_path):
        return False, "MS path missing"
    try:
        q = "QUERY scaninfo(MS2DATA)"
        res = msql_engine.process_query(q, ms_path, ms1_df=None, ms2_df=None, cache=None, parallel=False)
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

def _extract_ms2prod_targets_from_query(qtext: str):
    """
    Extract numeric MS2PROD targets and their local/global tolerances
    from a MassQL query string.

    Returns a list of dicts like:
    [{"target": 151.0, "tol": 1.0, "label": "151"}, ...]
    """
    out = []
    if not qtext:
        return out

    q = str(qtext)

    # global fallback tolerance
    global_tol = None
    m_global = re.search(r"TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)", q, flags=re.IGNORECASE)
    if m_global:
        try:
            global_tol = float(m_global.group(1))
        except Exception:
            global_tol = None

    # find each MS2PROD=(...)
    for m in re.finditer(r"MS2PROD\s*=\s*\(([^)]*)\)", q, flags=re.IGNORECASE):
        block = m.group(1)

        # local tolerance after this block, before next AND/OR/MS2...
        tail = q[m.end():]
        m_local = re.search(r"^\s*:\s*TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)", tail, flags=re.IGNORECASE)
        local_tol = None
        if m_local:
            try:
                local_tol = float(m_local.group(1))
            except Exception:
                local_tol = None

        tol = local_tol if local_tol is not None else global_tol
        if tol is None:
            tol = 0.02

        # split OR terms and keep only numeric entries
        parts = re.split(r"\bOR\b", block, flags=re.IGNORECASE)
        for part in parts:
            s = part.strip()
            try:
                val = float(s)
                out.append({
                    "target": val,
                    "tol": tol,
                    "label": s
                })
            except Exception:
                # ignore non-numeric targets such as aminoaciddelta(...)
                pass

    return out


def _build_query_peak_matches(mz, qdf: pd.DataFrame):
    """
    For a selected scan, collect all MS2PROD targets from matched queries
    and map them to actual peaks in the spectrum.
    """
    matches = []

    if qdf is None or qdf.empty or "query_text" not in qdf.columns:
        return matches

    qview = qdf.copy()
    keep_cols = [c for c in ["rule_name", "query_label", "query_text"] if c in qview.columns]
    qview = qview[keep_cols].drop_duplicates()

    for _, row in qview.iterrows():
        rule_name = str(row["rule_name"]) if "rule_name" in row and pd.notna(row["rule_name"]) else ""
        query_label = str(row["query_label"]) if "query_label" in row and pd.notna(row["query_label"]) else ""
        qtext = str(row["query_text"]) if pd.notna(row["query_text"]) else ""

        qname = rule_name.strip() or query_label.strip() or "matched_query"

        targets = _extract_ms2prod_targets_from_query(qtext)

        for t in targets:
            target = t["target"]
            tol = t["tol"]

            hit_idx = np.where(np.abs(mz - target) <= tol)[0]
            for idx in hit_idx:
                matches.append({
                    "query_name": qname,
                    "target": target,
                    "tol": tol,
                    "peak_mz": float(mz[idx]),
                    "peak_idx": int(idx),
                })

    return matches

# ============================================================
# Helper: convert MS1-only MGF → mzML for MassQL MS1 queries
# ============================================================
def convert_ms1_mgf_to_mzml(mgf_path: str) -> str:
    """
    Force-convert an uploaded MGF file to a simple mzML with one MS1 spectrum
    per MGF entry, so MassQL can query MS1DATA / MS1MZ.

    IMPORTANT:
    This does not verify whether the original MGF is truly MS1-only.
    It rewrites all spectra as ms level 1.
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
    st.sidebar.warning("Logo_massQL not found at static/logo_massQL.png")

try:
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")



st.sidebar.markdown("""by Ricardo M Borges (IPPN-UFRJ)""")

st.sidebar.markdown("""---""")


# ============================================================
# Parse compendiums
# ============================================================
QUERY_START_RE = re.compile(r'^\s*QUERY\b', flags=re.IGNORECASE)
_NUM = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

@dataclass
class MassQLRow:
    name: str
    query: str
    section: Optional[str] = None

SECTION_RE = re.compile(r"^#{1,}\s*(.+?)\s*$")  # e.g. "########### Benzoic Acids"

def parse_massql_compendium_text(text: str) -> Tuple[List[MassQLRow], List[str]]:
    """
    Parses a mixed compendium text file like your example:
    - Ignores blank lines
    - Tracks section headers like '########### Benzoic Acids'
    - Extracts rows of the form: <name>\t<query>
      (split ONLY on the first TAB)
    Returns (rows, warnings).
    """
    rows: List[MassQLRow] = []
    warnings: List[str] = []

    current_section: Optional[str] = None
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        # Section headers and comments
        m = SECTION_RE.match(line)
        if m:
            current_section = m.group(1).strip()
            continue
        if line.startswith("#"):
            continue

        # Must be TSV: name \t query
        if "\t" not in line:
            # Some people paste with multiple spaces instead of tabs; you *could* support that,
            # but better to warn so the file stays consistent.
            warnings.append(f"Line {i}: no TAB found, skipped -> {raw[:120]}")
            continue

        name, query = line.split("\t", 1)
        name = name.strip()
        query = query.strip()

        if not name or not query:
            warnings.append(f"Line {i}: empty name or query, skipped")
            continue

        # Light sanity check (optional)
        if "QUERY" not in query or "scaninfo" not in query:
            warnings.append(f"Line {i}: doesn't look like a MassQL query, kept anyway -> {name}")

        rows.append(MassQLRow(name=name, query=query, section=current_section))

    return rows, warnings

def parse_compendium_auto(text: str, fallback_name: str = "UploadedCompendium") -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Returns qitems in the SAME schema your runner expects:
    [{"section": "...", "query": "...", "name": "..."}]
    """
    warnings = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Heuristic: TSV format if many non-comment lines contain a TAB
    non_comment = [ln for ln in lines if not ln.startswith("#")]
    tab_lines = [ln for ln in non_comment if "\t" in ln]

    is_tsv_style = any(("\tQUERY " in ln.upper()) or ("\tQUERY\t" in ln.upper()) or ("\tQUERY" in ln.upper())
                   for ln in tab_lines) or (len(tab_lines) >= 1 and len(tab_lines) >= max(1, len(non_comment)//2))

    if is_tsv_style:
        rows, w = parse_massql_compendium_text(text)
        warnings.extend(w)

        qitems = []
        for r in rows:
            qitems.append({
                "section": r.section or "UnnamedSection",
                "query": r.query,
                "name": r.name,  # keep label if you want later
            })
        return qitems, warnings

    # Otherwise assume old style: multi-line QUERY blocks with # sections
    # We need a text-based version of parse_compendium() (yours is path-based)
    qitems = []
    current_section = None
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
            qitems.append({
                "section": current_section or "UnnamedSection",
                "query": qtext,
                "name": fallback_name,
            })
            continue

        i += 1

    if not qitems:
        warnings.append("No queries parsed (file might be empty or in an unexpected format).")

    return qitems, warnings

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
from typing import Optional

def run_compendiums(
    compendium_files: List[str],
    ms_files: List[str],
    *,
    parsed_compendia: Optional[List[Dict]] = None,
    use_loader_frames: bool = True,
    parallel: bool = False,
    qualifier_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    source_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      combined_unique: long table of hits (deduped)
      presence_by_comp: dict[compendium] -> presence matrix (scan-level)
      presence_global: presence matrix (compendium x section)
      query_registry: registry of executed queries (including 0-hit and errors)
    """

    def _get_ci(d: dict, key: str):
        """Case-insensitive dict key getter."""
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

    def _disp_name(path: str) -> str:
        if source_name_map:
            return source_name_map.get(path, os.path.basename(path))
        return os.path.basename(path)

    # ------------------------------------------------------------
    # Choose compendium iterator
    # ------------------------------------------------------------
    if parsed_compendia is not None:
        # expected: [{"comp_name": "...", "qitems": [{"section":..,"query":..,"name":..}, ...]}, ...]
        comp_iter = parsed_compendia
    else:
        comp_iter = [{"comp_name": Path(p).stem, "path": p, "qitems": None} for p in compendium_files]

    all_hits: List[pd.DataFrame] = []
    query_rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------
    # Helpers: execute query with registry logging
    # ------------------------------------------------------------
    def _exec_query(
        *,
        base_row: Dict[str, object],
        label: str,
        q: str,
        or_patch: bool,
        ms_path: str,
        ms1_df: Optional[pd.DataFrame],
        ms2_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Execute a MassQL query and always register status/n_hits/error."""
        query_rows.append({**base_row, "query_label": label, "query_text": q, "or_patch": or_patch})
        try:
            r = msql_engine.process_query(
                q,
                ms_path,
                ms1_df=ms1_df,
                ms2_df=ms2_df,
                cache=True,
                parallel=parallel,
            )
            nh = int(len(r)) if isinstance(r, pd.DataFrame) else 0
            query_rows[-1]["status"] = "ok"
            query_rows[-1]["n_hits"] = nh
            return r if isinstance(r, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            query_rows[-1]["status"] = "error"
            query_rows[-1]["error"] = str(e)
            query_rows[-1]["n_hits"] = 0
            return pd.DataFrame()


    # ------------------------------------------------------------
    # Main loop: MS files -> compendia -> queries
    # ------------------------------------------------------------
    for ms_path in ms_files:
        ms_path = os.fspath(ms_path)
        ext = Path(ms_path).suffix.lower()

        # --- canonical map (scan → precmz / rt / NAME / SPECTRUMID) ---
        ms_rows = []
        with _open_ms(ms_path) as _rdr_ms_:
            for _i, _spec in enumerate(_rdr_ms_, start=1):
                _scan, _pepmz, _rtsec = _extract_meta_from_spec(_spec, _i, ext)

                _nm = None
                _sid = None
                if ext == ".mgf":
                    P = _spec.get("params", {}) or {}
                    _nm = _get_ci(P, "NAME")
                    _sid = _get_ci(P, "SPECTRUMID")

                ms_rows.append(
                    {
                        "scan": _scan,
                        "precmz_mgf": _pepmz,
                        "rt_mgf": _rtsec,
                        "NAME": _nm,
                        "SPECTRUMID": _sid,
                    }
                )

        ms_map = pd.DataFrame(ms_rows)
        if not ms_map.empty:
            ms_map["scan"] = pd.to_numeric(ms_map["scan"], errors="coerce").astype("Int64")

        # --- optional loader frames + alignment ---
        ms1_df = None
        ms2_df = None
        if use_loader_frames:
            try:
                ms1_df, ms2_df = msql_fileloading.load_data(ms_path)
            except Exception as e:
                st.warning(f"[WARN] load_data failed for {os.path.basename(ms_path)}: {e}")
                ms1_df, ms2_df = None, None
            if ms2_df is not None:
                ms2_df = align_ms2_with_mgf(ms2_df, ms_path)

        # --- iterate compendiums/queries ---
        for comp_pack in comp_iter:
            comp_name = comp_pack.get("comp_name") or "UnnamedCompendium"

            # get qitems either from parsed_compendia, or from disk path
            if comp_pack.get("qitems") is not None:
                qitems = comp_pack["qitems"]
            else:
                comp_file = comp_pack.get("path")
                if not comp_file:
                    st.warning(f"[WARN] Missing compendium path for {comp_name}; skipping.")
                    continue
                try:
                    raw = Path(comp_file).read_text(encoding="utf-8", errors="replace")
                    qitems, warns = parse_compendium_auto(raw, fallback_name=comp_name)
                    if warns:
                        st.warning(f"[WARN] {comp_name}: " + " | ".join(warns[:5]))
                except Exception as e:
                    st.warning(f"[WARN] Failed to parse {comp_file}: {e}")
                    continue

            comp_over = _select_overrides(qualifier_overrides, comp_name)

            for q_idx, item in enumerate(qitems, start=1):
                qtext = item.get("query", "") or ""
                section = item.get("section") or "UnnamedSection"
                rule_name = item.get("name") or f"#{q_idx}"

                if comp_over:
                    qtext = apply_qualifier_overrides(qtext, comp_over)

                base_query_label = f"{comp_name} :: {section} :: {rule_name}"

                base_row = {
                    "source_file": _disp_name(ms_path),
                    "ms_path": ms_path,
                    "compendium": comp_name,
                    "section": section,
                    "query_idx": q_idx,
                    "rule_name": rule_name,
                }

                # Execute query exactly as written
                ####################################
                res = _exec_query(
                    base_row=base_row,
                    label=base_query_label,
                    q=qtext,
                    or_patch=False,
                    ms_path=ms_path,
                    ms1_df=ms1_df,
                    ms2_df=ms2_df,
                )


                # --- normalize schema ---
                if res is None or not isinstance(res, pd.DataFrame) or res.empty:
                    continue

                res = res.copy()

                # ensure query_label/query_text present in result rows
                if "query_label" not in res.columns:
                    res["query_label"] = base_query_label
                if "query_text" not in res.columns:
                    res["query_text"] = qtext

                # unify scan column
                scan_variants = [c for c in res.columns if str(c).lower() in ("scan", "scans")]
                if scan_variants:
                    src = scan_variants[0]
                    res["scan"] = res[src]
                elif any(c.lower() == "spectrumindex" for c in res.columns):
                    true_col = [c for c in res.columns if c.lower() == "spectrumindex"][0]
                    res["scan"] = pd.to_numeric(res[true_col], errors="coerce").add(1)
                elif any(c.lower() == "spectrum_id" for c in res.columns):
                    true_col = [c for c in res.columns if c.lower() == "spectrum_id"][0]
                    res["scan"] = pd.to_numeric(res[true_col], errors="coerce")

                # fill expected columns
                for col in ("precmz", "rt", "compendium", "section", "query_idx", "source_file", "rule_name"):
                    if col not in res.columns:
                        res[col] = pd.NA

                # overwrite missing-only columns with known values
                if res["compendium"].isna().all():
                    res["compendium"] = comp_name
                if res["section"].isna().all():
                    res["section"] = section
                if res["query_idx"].isna().all():
                    res["query_idx"] = q_idx
                if res["rule_name"].isna().all():
                    res["rule_name"] = rule_name

                disp = _disp_name(ms_path)
                if res["source_file"].isna().all():
                    res["source_file"] = disp

                # uppercase fallbacks
                for c in ("precmz", "rt"):
                    if c not in res.columns and c.upper() in res.columns:
                        res[c] = res[c.upper()]

                # numeric coercions
                if "scan" in res.columns:
                    res["scan"] = pd.to_numeric(res["scan"], errors="coerce")
                for c in ("precmz", "rt"):
                    if c in res.columns:
                        res[c] = pd.to_numeric(res[c], errors="coerce")

                # --- backfill from canonical map ---
                if "scan" in res.columns and not ms_map.empty:
                    res = res.merge(ms_map, on="scan", how="left")

                    if "precmz" in res.columns:
                        res["precmz"] = res["precmz"].fillna(res.get("precmz_mgf"))
                    else:
                        res["precmz"] = res.get("precmz_mgf")

                    if "rt" in res.columns:
                        res["rt"] = res["rt"].fillna(res.get("rt_mgf"))
                    else:
                        res["rt"] = res.get("rt_mgf")

                    res = res.drop(columns=["precmz_mgf", "rt_mgf"], errors="ignore")

                all_hits.append(res)

    # ------------------------------------------------------------
    # Query registry dataframe
    # ------------------------------------------------------------
    query_registry = pd.DataFrame(query_rows)
    if not query_registry.empty:
        for c in ("status", "error", "n_hits", "or_patch"):
            if c not in query_registry.columns:
                query_registry[c] = pd.NA

    # ------------------------------------------------------------
    # No hits at all
    # ------------------------------------------------------------
    if not all_hits:
        empty = pd.DataFrame(
            columns=[
                "scan",
                "precmz",
                "rt",
                "compendium",
                "section",
                "query_idx",
                "source_file",
                "NAME",
                "SPECTRUMID",
                "query_label",
                "query_text",
            ]
        )
        return empty, {}, empty, query_registry

    # ------------------------------------------------------------
    # Combine + dedupe
    # ------------------------------------------------------------
    combined = pd.concat(all_hits, ignore_index=True)

    dedup_keys = ["source_file", "compendium", "section", "rule_name", "query_idx", "query_label", "scan"]
    for k in dedup_keys:
        if k not in combined.columns:
            combined[k] = pd.NA

    combined_unique = combined.drop_duplicates(subset=dedup_keys).reset_index(drop=True)

    # ------------------------------------------------------------
    # Presence matrices
    # ------------------------------------------------------------
    presence_global = pd.DataFrame()
    presence_by_comp: Dict[str, pd.DataFrame] = {}

    need_cols = {"source_file", "scan", "compendium", "section"}
    if need_cols.issubset(set(combined_unique.columns)):
        dfp = combined_unique.copy()
        dfp["source_file"] = dfp["source_file"].astype(str)
        dfp["scan"] = pd.to_numeric(dfp["scan"], errors="coerce")
        dfp["compendium"] = dfp["compendium"].astype(str)
        dfp["section"] = dfp["section"].astype(str)
        dfp = dfp.dropna(subset=["source_file", "scan", "compendium", "section"])

        if not dfp.empty:
            dfp["hit"] = 1

            presence_global = (
                dfp.pivot_table(
                    index=["source_file", "scan"],
                    columns=["compendium", "section"],
                    values="hit",
                    aggfunc="max",
                    fill_value=0,
                )
                .sort_index()
            )

            for comp in sorted(dfp["compendium"].unique()):
                sub = dfp.loc[dfp["compendium"] == comp].copy()
                if sub.empty:
                    continue
                pres = (
                    sub.pivot_table(
                        index=["source_file", "scan"],
                        columns="section",
                        values="hit",
                        aggfunc="max",
                        fill_value=0,
                    )
                    .sort_index()
                )
                presence_by_comp[comp] = pres

    return combined_unique, presence_by_comp, presence_global, query_registry
    
    
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


def _extract_first_float(text: str):
    m = re.search(r"([0-9]*\.?[0-9]+)", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_global_tolerancemz(qtext: str, default: float = 0.02) -> float:
    m = re.search(r"TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)", str(qtext), flags=re.IGNORECASE)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def _extract_ms2prod_targets(qtext: str):
    """
    Returns:
    [
        {"target": 151.0, "tol": 1.0},
        {"target": 179.0, "tol": 1.0},
    ]
    """
    out = []
    q = str(qtext)
    global_tol = _extract_global_tolerancemz(q, default=0.02)

    for m in re.finditer(r"MS2PROD\s*=\s*\(([^)]*)\)", q, flags=re.IGNORECASE):
        block = m.group(1)
        tail = q[m.end():]

        m_local = re.search(r"^\s*:\s*TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)", tail, flags=re.IGNORECASE)
        tol = global_tol
        if m_local:
            try:
                tol = float(m_local.group(1))
            except Exception:
                pass

        parts = re.split(r"\bOR\b", block, flags=re.IGNORECASE)
        for part in parts:
            s = part.strip()
            try:
                out.append({"target": float(s), "tol": tol})
            except Exception:
                pass

    return out


def _extract_ms2nl_targets(qtext: str):
    """
    Returns:
    [
        {"target": 162.05282, "tol": 0.02},
    ]
    """
    out = []
    q = str(qtext)
    global_tol = _extract_global_tolerancemz(q, default=0.02)

    for m in re.finditer(r"MS2NL\s*=\s*\(([^)]*)\)", q, flags=re.IGNORECASE):
        block = m.group(1).strip()
        tail = q[m.end():]

        m_local = re.search(r"^\s*:\s*TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)", tail, flags=re.IGNORECASE)
        tol = global_tol
        if m_local:
            try:
                tol = float(m_local.group(1))
            except Exception:
                pass

        try:
            out.append({"target": float(block), "tol": tol})
        except Exception:
            pass

    return out


def _find_ms2prod_peak_matches(mz: np.ndarray, y: np.ndarray, qdf: pd.DataFrame):
    matches = []
    if qdf is None or qdf.empty or "query_text" not in qdf.columns:
        return matches

    qview = qdf.drop_duplicates(subset=["query_text"]).copy()

    for _, row in qview.iterrows():
        qtext = str(row["query_text"])
        targets = _extract_ms2prod_targets(qtext)

        for t in targets:
            target = t["target"]
            tol = t["tol"]

            idxs = np.where(np.abs(mz - target) <= tol)[0]
            if len(idxs) == 0:
                continue

            # choose closest peak
            best = idxs[np.argmin(np.abs(mz[idxs] - target))]
            matches.append({
                "kind": "MS2PROD",
                "target": target,
                "tol": tol,
                "peak_idx": int(best),
                "peak_mz": float(mz[best]),
                "peak_y": float(y[best]),
            })

    return matches


def _find_ms2nl_peak_matches(mz: np.ndarray, y: np.ndarray, qdf: pd.DataFrame):
    matches = []
    if qdf is None or qdf.empty or "query_text" not in qdf.columns:
        return matches

    qview = qdf.drop_duplicates(subset=["query_text"]).copy()

    for _, row in qview.iterrows():
        qtext = str(row["query_text"])
        targets = _extract_ms2nl_targets(qtext)

        for t in targets:
            target = t["target"]
            tol = t["tol"]

            best_pair = None
            best_err = None

            for i in range(len(mz)):
                for j in range(i + 1, len(mz)):
                    diff = abs(float(mz[j]) - float(mz[i]))
                    err = abs(diff - target)
                    if err <= tol:
                        if best_err is None or err < best_err:
                            best_err = err
                            best_pair = (i, j, diff)

            if best_pair is not None:
                i, j, diff = best_pair
                matches.append({
                    "kind": "MS2NL",
                    "target": target,
                    "tol": tol,
                    "i": int(i),
                    "j": int(j),
                    "mz_i": float(mz[i]),
                    "mz_j": float(mz[j]),
                    "y_i": float(y[i]),
                    "y_j": float(y[j]),
                    "diff": float(diff),
                })

    return matches


def _plot_ms2_html_from_index(
    _MGF: dict,
    scan: int,
    normalize: bool = True,
    annotate_top_n: int = 12,
    matched_queries_df: pd.DataFrame | None = None,
) -> str:
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

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Base spectrum
    for x, h in zip(mz, y):
        ax.vlines(x, 0.0, h, linewidth=2.0)

    pts = ax.scatter(mz, y, s=10, alpha=0)
    labels = [
        f"m/z: {x:.4f}<br>I: {h:.2f}{'%' if normalize else ''}"
        for x, h in zip(mz, y)
    ]
    tooltip = plugins.PointHTMLTooltip(pts, labels=labels, hoffset=10, voffset=-10)
    plugins.connect(fig, tooltip, plugins.Reset(), plugins.Zoom(), plugins.BoxZoom())

    # Precursor
    pep = P.get("pepmass")
    pepmz = pep[0] if isinstance(pep, (list, tuple, np.ndarray)) else pep
    try:
        pepmz = float(pepmz)
    except Exception:
        pepmz = None

    if pepmz is not None and not math.isnan(pepmz):
        ax.axvline(pepmz, linestyle="--", linewidth=1.0)
        ax.text(
            pepmz,
            ax.get_ylim()[1] * 0.95,
            f"precursor {pepmz:.4f}",
            rotation=90,
            va="top",
            ha="right",
        )

    # Top-N labels
    if annotate_top_n and annotate_top_n > 0:
        idx = np.argsort(y)[-annotate_top_n:]
        ymax = ax.get_ylim()[1] or 1
        for k in idx:
            ax.text(
                mz[k],
                y[k] + 0.02 * ymax,
                f"{mz[k]:.4f}",
                rotation=90,
                va="bottom",
                ha="center",
            )

    prod_matches = []
    nl_matches = []

    if matched_queries_df is not None and not matched_queries_df.empty and "query_text" in matched_queries_df.columns:
        qview = matched_queries_df.drop_duplicates(subset=["query_text"]).copy()

        for _, row in qview.iterrows():
            qtext = str(row["query_text"])

            # -------------------------
            # MS2PROD
            # -------------------------
            global_tol = 0.02
            m_global = re.search(
                r"TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)",
                qtext,
                flags=re.IGNORECASE,
            )
            if m_global:
                try:
                    global_tol = float(m_global.group(1))
                except Exception:
                    pass

            for m in re.finditer(r"MS2PROD\s*=\s*\(([^)]*)\)", qtext, flags=re.IGNORECASE):
                block = m.group(1)
                tail = qtext[m.end():]

                tol = global_tol
                m_local = re.search(
                    r"^\s*:\s*TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)",
                    tail,
                    flags=re.IGNORECASE,
                )
                if m_local:
                    try:
                        tol = float(m_local.group(1))
                    except Exception:
                        pass

                parts = re.split(r"\bOR\b", block, flags=re.IGNORECASE)
                for part in parts:
                    s = part.strip()
                    try:
                        target = float(s)
                    except Exception:
                        continue

                    idxs = np.where(np.abs(mz - target) <= tol)[0]
                    if len(idxs) == 0:
                        continue

                    best = idxs[np.argmin(np.abs(mz[idxs] - target))]
                    prod_matches.append({
                        "target": target,
                        "tol": tol,
                        "peak_idx": int(best),
                        "peak_mz": float(mz[best]),
                        "peak_y": float(y[best]),
                    })

            # -------------------------
            # MS2NL
            # -------------------------
            for m in re.finditer(r"MS2NL\s*=\s*\(([^)]*)\)", qtext, flags=re.IGNORECASE):
                block = m.group(1).strip()
                tail = qtext[m.end():]

                tol = global_tol
                m_local = re.search(
                    r"^\s*:\s*TOLERANCEMZ\s*=\s*([0-9]*\.?[0-9]+)",
                    tail,
                    flags=re.IGNORECASE,
                )
                if m_local:
                    try:
                        tol = float(m_local.group(1))
                    except Exception:
                        pass

                try:
                    target = float(block)
                except Exception:
                    continue

                best_pair = None
                best_err = None

                for i in range(len(mz)):
                    for j in range(i + 1, len(mz)):
                        diff = abs(float(mz[j]) - float(mz[i]))
                        err = abs(diff - target)
                        if err <= tol:
                            if best_err is None or err < best_err:
                                best_err = err
                                best_pair = (i, j, diff)

                if best_pair is not None:
                    i, j, diff = best_pair
                    nl_matches.append({
                        "target": target,
                        "tol": tol,
                        "i": int(i),
                        "j": int(j),
                        "mz_i": float(mz[i]),
                        "mz_j": float(mz[j]),
                        "y_i": float(y[i]),
                        "y_j": float(y[j]),
                        "diff": float(diff),
                    })

    # Draw MS2PROD matches
    if prod_matches:
        ymax = ax.get_ylim()[1] or 1
        done = set()

        for pm in prod_matches:
            x = pm["peak_mz"]
            h = pm["peak_y"]
            target = pm["target"]

            key = (round(x, 4), round(target, 4))
            if key in done:
                continue

            ax.scatter([x], [h], s=70, marker="o", zorder=6)
            ax.text(
                x,
                h + 0.05 * ymax,
                f"PROD {target:.4f}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=8,
            )
            done.add(key)

    # Draw MS2NL matches
    if nl_matches:
        ymax = ax.get_ylim()[1] or 1
        used = set()

        for nm in nl_matches:
            x1 = nm["mz_i"]
            x2 = nm["mz_j"]
            y1 = nm["y_i"]
            y2 = nm["y_j"]
            d = nm["target"]

            key = (round(x1, 4), round(x2, 4), round(d, 4))
            if key in used:
                continue

            y_arrow = max(y1, y2) + 0.10 * ymax

            ax.plot([x1, x2], [y_arrow, y_arrow], linewidth=1.8)
            ax.scatter([x1, x2], [y1, y2], s=45, marker="s", zorder=6)

            ax.text(
                (x1 + x2) / 2.0,
                y_arrow + 0.03 * ymax,
                f"NL {d:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            used.add(key)

    # Debug box
    dbg = f"matched PROD marks: {len(prod_matches)}\nmatched NL marks: {len(nl_matches)}"
    ax.text(
        0.01,
        0.99,
        dbg,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # RT
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

    html = mpld3.fig_to_html(fig, no_extras=False)
    plt.close(fig)
    return html

def _save_html_text(html_text: str, out_path: str):
    Path(out_path).write_text(html_text, encoding="utf-8")

def _plot_ms2_diagnostic_pair_html(
    _MGF: dict,
    scan: int,
    diag_row: dict,
    normalize: bool = True,
    annotate_top_n: int = 12,
    rectangle_pad_x: float = 0.15,
    rectangle_height_frac: float = 0.10,
) -> str:
    try:
        import mpld3
        from mpld3 import plugins
        from matplotlib.patches import Rectangle
    except Exception as e:
        return f"<div style='color:#b00'>mpld3/matplotlib patch support not available: {e}</div>"

    rec = _MGF.get(int(scan))
    if rec is None or rec["mz"].size == 0 or rec["i"].size == 0:
        return f"<div style='color:#b00'>No peaks found for scan {scan}</div>"

    mz = np.asarray(rec["mz"], dtype=float)
    I = np.asarray(rec["i"], dtype=float)
    P = rec["params"]

    y = I.astype(float)
    if normalize and y.size > 0 and y.max() > 0:
        y = (y / y.max()) * 100.0

    x_mz = float(diag_row["x_mz"])
    y_mz = float(diag_row["y_mz"])
    expected_delta = float(diag_row["expected_delta"])
    observed_delta = float(diag_row["observed_delta"])
    error_da = float(diag_row["error_da"])
    residue = str(diag_row.get("residue", ""))

    # nearest peak indices
    idx_x = int(np.argmin(np.abs(mz - x_mz)))
    idx_y = int(np.argmin(np.abs(mz - y_mz)))

    peak_x = float(mz[idx_x])
    peak_y = float(mz[idx_y])
    int_x = float(y[idx_x])
    int_y = float(y[idx_y])

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # full spectrum
    for xx, hh in zip(mz, y):
        ax.vlines(xx, 0.0, hh, linewidth=2.0)

    pts = ax.scatter(mz, y, s=10, alpha=0)
    labels = [
        f"m/z: {xx:.4f}<br>I: {hh:.2f}{'%' if normalize else ''}"
        for xx, hh in zip(mz, y)
    ]
    tooltip = plugins.PointHTMLTooltip(pts, labels=labels, hoffset=10, voffset=-10)
    plugins.connect(fig, tooltip, plugins.Reset(), plugins.Zoom(), plugins.BoxZoom())

    # precursor
    pep = P.get("pepmass")
    pepmz = pep[0] if isinstance(pep, (list, tuple, np.ndarray)) else pep
    try:
        pepmz = float(pepmz)
    except Exception:
        pepmz = None

    if pepmz is not None and not math.isnan(pepmz):
        ax.axvline(pepmz, linestyle="--", linewidth=1.0)
        ax.text(
            pepmz,
            ax.get_ylim()[1] * 0.95,
            f"precursor {pepmz:.4f}",
            rotation=90,
            va="top",
            ha="right",
        )

    # top-N labels
    if annotate_top_n and annotate_top_n > 0:
        idx_top = np.argsort(y)[-annotate_top_n:]
        ymax = ax.get_ylim()[1] or 1.0
        for k in idx_top:
            ax.text(
                mz[k],
                y[k] + 0.02 * ymax,
                f"{mz[k]:.4f}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=8,
            )

    ymax = ax.get_ylim()[1] or 1.0
    rect_h_x = max(int_x * 1.05, rectangle_height_frac * ymax)
    rect_h_y = max(int_y * 1.05, rectangle_height_frac * ymax)

    # rectangles around matched peaks
    rect1 = Rectangle(
        (peak_x - rectangle_pad_x, 0.0),
        2 * rectangle_pad_x,
        rect_h_x,
        fill=False,
        linewidth=2.2,
        zorder=7,
    )
    rect2 = Rectangle(
        (peak_y - rectangle_pad_x, 0.0),
        2 * rectangle_pad_x,
        rect_h_y,
        fill=False,
        linewidth=2.2,
        zorder=7,
    )
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # emphasize matched peaks
    ax.scatter([peak_x], [int_x], s=80, marker="s", zorder=8)
    ax.scatter([peak_y], [int_y], s=80, marker="s", zorder=8)

    # labels over the peaks
    ax.text(
        peak_x,
        rect_h_x + 0.03 * ymax,
        f"X\n{peak_x:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.text(
        peak_y,
        rect_h_y + 0.03 * ymax,
        f"X+Δ\n{peak_y:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # connector / annotation
    y_line = max(rect_h_x, rect_h_y) + 0.10 * ymax
    ax.plot([peak_x, peak_y], [y_line, y_line], linewidth=1.8, zorder=6)

    label = (
        f"Δ {residue}" # expected={expected_delta:.5f}\n"
        f"observed={observed_delta:.5f} | error={error_da:+.5f}")
    ax.text(
        (peak_x + peak_y) / 2.0,
        y_line + 0.03 * ymax,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.85),
    )

    # RT
    rt = P.get("rtinseconds")
    try:
        rt = float(rt) if rt is not None else None
    except Exception:
        rt = None

    title_rt = f" | RT={rt:.1f}s" if rt is not None else ""
    title_prec = f" | Precursor={pepmz:.4f}" if pepmz is not None else ""
    ax.set_title(f"Diagnostic pair | Scan {scan}{title_rt}{title_prec}")
    ax.set_xlabel("m/z")
    ax.set_ylabel(f"Intensity ({'relative %' if normalize else 'a.u.'})")

    dbg = (
        f"x_mz={peak_x:.5f}\n"
        f"y_mz={peak_y:.5f}\n"
        f"expected_delta={expected_delta:.5f}\n"
        f"observed_delta={observed_delta:.5f}"
    )
    ax.text(
        0.01,
        0.99,
        dbg,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    html = mpld3.fig_to_html(fig, no_extras=False)
    plt.close(fig)
    return html

# ============================================================
# TAB Diagnostics
# ============================================================
AA_DELTA = {
    "G": 57.02146,
    "A": 71.03711,
    "S": 87.03203,
    "P": 97.05276,
    "V": 99.06841,
    "T": 101.04768,
    "C": 103.00919,
    "L": 113.08406,
    "I": 113.08406,
    "N": 114.04293,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04049,
    "H": 137.05891,
    "F": 147.06841,
    "R": 156.10111,
    "Y": 163.06333,
    "W": 186.07931,
}

def _extract_intensitypercent(qtext: str, default: float = 0.0) -> float:
    m = re.search(r"INTENSITYPERCENT\s*=\s*([0-9]*\.?[0-9]+)", str(qtext), flags=re.IGNORECASE)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default

def _parse_x_plus_delta_query(qtext: str):
    q = str(qtext)

    if not re.search(r"MS2PROD\s*=\s*X\b", q, flags=re.IGNORECASE):
        return None

    m = re.search(
        r"MS2PROD\s*=\s*\(\s*X\s*\+\s*aminoaciddelta\(\s*([A-Za-z]+)\s*\)\s*\)",
        q,
        flags=re.IGNORECASE
    )
    if not m:
        return None

    residue = m.group(1).strip()
    if residue not in AA_DELTA:
        return None

    tol = _extract_global_tolerancemz(q, default=0.02)
    min_int_pct = _extract_intensitypercent(q, default=0.0)

    return {
        "mode": "x_plus_aminoaciddelta",
        "residue": residue,
        "delta": AA_DELTA[residue],
        "tol": tol,
        "min_int_pct": min_int_pct,
        "query_text": q,
    }

def _parse_fixed_prod_query(qtext: str):
    """
    Parse numeric MS2PROD targets from queries like:
    MS2PROD=30.0344
    MS2PROD=30.0344 AND MS2PROD=44.0500

    Returns a list of float targets, or None if none are found.
    """
    q = str(qtext)

    matches = re.findall(
        r"MS2PROD\s*=\s*([0-9]*\.?[0-9]+)",
        q,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    try:
        vals = [float(m) for m in matches]
    except Exception:
        return None

    return vals if vals else None

def diagnose_delta_pairs_for_scan(mz: np.ndarray, y_rel: np.ndarray, qtext: str):
    spec = _parse_x_plus_delta_query(qtext)
    if spec is None:
        return []

    delta = spec["delta"]
    tol = spec["tol"]
    min_int_pct = spec["min_int_pct"]
    residue = spec["residue"]

    hits = []
    n = len(mz)

    for i in range(n):
        if y_rel[i] < min_int_pct:
            continue

        target = float(mz[i]) + delta
        idxs = np.where(np.abs(mz - target) <= tol)[0]

        for j in idxs:
            if i == j:
                continue
            if y_rel[j] < min_int_pct:
                continue

            obs_delta = float(mz[j]) - float(mz[i])
            err = obs_delta - delta

            hits.append({
                "mode": "x_plus_aminoaciddelta",
                "residue": residue,
                "x_mz": float(mz[i]),
                "y_mz": float(mz[j]),
                "expected_delta": float(delta),
                "observed_delta": float(obs_delta),
                "error_da": float(err),
                "intensity_x_pct": float(y_rel[i]),
                "intensity_y_pct": float(y_rel[j]),
                "tol": float(tol),
            })

    hits = sorted(hits, key=lambda d: abs(d["error_da"]))
    return hits

def _plot_ms2_diagnostic_html(
    _MGF: dict,
    scan: int,
    diag_row: dict,
    normalize: bool = True,
    annotate_top_n: int = 12,
    rectangle_pad_x: float = 0.15,
    rectangle_height_frac: float = 0.10,
) -> str:
    try:
        import mpld3
        from mpld3 import plugins
        from matplotlib.patches import Rectangle
    except Exception as e:
        return f"<div style='color:#b00'>mpld3/matplotlib patch support not available: {e}</div>"

    rec = _MGF.get(int(scan))
    if rec is None or rec["mz"].size == 0 or rec["i"].size == 0:
        return f"<div style='color:#b00'>No peaks found for scan {scan}</div>"

    mz = np.asarray(rec["mz"], dtype=float)
    I = np.asarray(rec["i"], dtype=float)
    P = rec["params"]

    y = I.astype(float)
    if normalize and y.size > 0 and y.max() > 0:
        y = (y / y.max()) * 100.0

    mode = str(diag_row.get("mode", ""))

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # full spectrum
    for xx, hh in zip(mz, y):
        ax.vlines(xx, 0.0, hh, linewidth=2.0)

    pts = ax.scatter(mz, y, s=10, alpha=0)
    labels = [
        f"m/z: {xx:.4f}<br>I: {hh:.2f}{'%' if normalize else ''}"
        for xx, hh in zip(mz, y)
    ]
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
        ax.text(
            pepmz,
            ax.get_ylim()[1] * 0.95,
            f"precursor {pepmz:.4f}",
            rotation=90,
            va="top",
            ha="right",
        )

    if annotate_top_n and annotate_top_n > 0:
        idx_top = np.argsort(y)[-annotate_top_n:]
        ymax0 = ax.get_ylim()[1] or 1.0
        for k in idx_top:
            ax.text(
                mz[k],
                y[k] + 0.02 * ymax0,
                f"{mz[k]:.4f}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=8,
            )

    ymax = ax.get_ylim()[1] or 1.0

    # -------------------------------------------------
    # MODE 1: fixed_prod
    # -------------------------------------------------
    if mode == "fixed_prod":
        prod_target = float(diag_row["prod_target"])
        peak_mz = float(diag_row["peak_mz"])
        error_da = float(diag_row["error_da"])
        intensity_pct = float(diag_row.get("intensity_pct", 0.0))

        idx_peak = int(np.argmin(np.abs(mz - peak_mz)))
        peak_real = float(mz[idx_peak])
        peak_y = float(y[idx_peak])

        rect_h = max(peak_y * 1.05, rectangle_height_frac * ymax)

        rect = Rectangle(
            (peak_real - rectangle_pad_x, 0.0),
            2 * rectangle_pad_x,
            rect_h,
            fill=False,
            linewidth=2.2,
            edgecolor="red",
            zorder=7,
        )
        ax.add_patch(rect)

        ax.scatter([peak_real], [peak_y], s=80, marker="s", zorder=8)

        ax.text(
            peak_real,
            rect_h + 0.03 * ymax,
            f"PROD\n{peak_real:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        label = (
            f"target={prod_target:.5f}\n"
            f"observed={peak_real:.5f} | error={error_da:+.5f}\n"
            f"intensity={intensity_pct:.2f}%"
        )

        ax.text(
            peak_real,
            rect_h + 0.12 * ymax,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.85),
        )

        dbg = (
            f"mode=fixed_prod\n"
            f"target={prod_target:.5f}\n"
            f"peak={peak_real:.5f}\n"
            f"error={error_da:+.5f}"
        )

    # -------------------------------------------------
    # MODE 2: x_plus_aminoaciddelta
    # -------------------------------------------------
    else:
        x_mz = float(diag_row["x_mz"])
        y_mz = float(diag_row["y_mz"])
        expected_delta = float(diag_row["expected_delta"])
        observed_delta = float(diag_row["observed_delta"])
        error_da = float(diag_row["error_da"])
        residue = str(diag_row.get("residue", ""))

        idx_x = int(np.argmin(np.abs(mz - x_mz)))
        idx_y = int(np.argmin(np.abs(mz - y_mz)))

        peak_x = float(mz[idx_x])
        peak_y = float(mz[idx_y])
        int_x = float(y[idx_x])
        int_y = float(y[idx_y])

        rect_h_x = max(int_x * 1.05, rectangle_height_frac * ymax)
        rect_h_y = max(int_y * 1.05, rectangle_height_frac * ymax)

        rect1 = Rectangle(
            (peak_x - rectangle_pad_x, 0.0),
            2 * rectangle_pad_x,
            rect_h_x,
            fill=False,
            linewidth=2.2,
            edgecolor="red",
            zorder=7,
        )
        rect2 = Rectangle(
            (peak_y - rectangle_pad_x, 0.0),
            2 * rectangle_pad_x,
            rect_h_y,
            fill=False,
            linewidth=2.2,
            edgecolor="red",
            zorder=7,
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.scatter([peak_x], [int_x], s=80, marker="s", zorder=8)
        ax.scatter([peak_y], [int_y], s=80, marker="s", zorder=8)

        ax.text(
            peak_x,
            rect_h_x + 0.03 * ymax,
            f"X\n{peak_x:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            peak_y,
            rect_h_y + 0.03 * ymax,
            f"X+Δ\n{peak_y:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        y_line = max(rect_h_x, rect_h_y) +  ymax * 0.08
        ax.plot([peak_x, peak_y], [y_line, y_line], linewidth=1.8, zorder=6, color="red")

        label = (
            f"{residue} Δ expected={expected_delta:.5f}\n"
            f"observed={observed_delta:.5f} | error={error_da:+.5f}"
        )

        ax.text(
            (peak_x + peak_y) / 2.0,
            y_line + 0.03 * ymax,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.85),
        )

        dbg = (
            f"mode=x_plus_aminoaciddelta\n"
            f"x_mz={peak_x:.5f}\n"
            f"y_mz={peak_y:.5f}\n"
            f"expected_delta={expected_delta:.5f}\n"
            f"observed_delta={observed_delta:.5f}"
        )

    ax.text(
        0.01,
        0.99,
        dbg,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    rt = P.get("rtinseconds")
    try:
        rt = float(rt) if rt is not None else None
    except Exception:
        rt = None

    title_rt = f" | RT={rt:.1f}s" if rt is not None else ""
    title_prec = f" | Precursor={pepmz:.4f}" if pepmz is not None else ""

    ax.set_title(f"Diagnostic plot | Scan {scan}{title_rt}{title_prec}")
    ax.set_xlabel("m/z")
    ax.set_ylabel(f"Intensity ({'relative %' if normalize else 'a.u.'})")

    fig.tight_layout()
    html = mpld3.fig_to_html(fig, no_extras=False)
    plt.close(fig)
    return html

def run_diagnostics_from_matches(combined_df: pd.DataFrame, ms_indexes: dict, display_to_view_path: dict):
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    need = {"source_file", "scan", "query_text"}
    if not need.issubset(combined_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    work = (
        combined_df[
            ["source_file", "scan", "query_text", "query_label", "rule_name", "compendium", "section"]
        ]
        .dropna(subset=["source_file", "scan", "query_text"])
        .drop_duplicates()
        .copy()
    )

    diag_rows = []
    pair_rows = []

    for _, row in work.iterrows():
        source_file = str(row["source_file"])
        scan_val = pd.to_numeric(row["scan"], errors="coerce")
        if pd.isna(scan_val):
            continue
        scan = int(scan_val)

        qtext = str(row["query_text"])
        query_label = row.get("query_label", "")
        rule_name = row.get("rule_name", "")
        compendium = row.get("compendium", "")
        section = row.get("section", "")

        ms_path = display_to_view_path.get(source_file)
        if not ms_path:
            continue

        ms_index = ms_indexes.get(ms_path, {})
        rec = ms_index.get(scan)
        if rec is None:
            continue

        mz = np.asarray(rec["mz"], dtype=float)
        I = np.asarray(rec["i"], dtype=float)

        if mz.size == 0 or I.size == 0:
            continue

        y_rel = I.astype(float)
        if y_rel.max() > 0:
            y_rel = (y_rel / y_rel.max()) * 100.0

        # -------------------------------------------------
        # Parse supported diagnostic modes
        # -------------------------------------------------
        parsed_delta = _parse_x_plus_delta_query(qtext)
        fixed_targets = None if parsed_delta is not None else _parse_fixed_prod_query(qtext)

        if parsed_delta is not None:
            diagnostic_mode = parsed_delta["mode"]
            supported = True
        elif fixed_targets is not None:
            diagnostic_mode = "fixed_prod"
            supported = True
        else:
            diagnostic_mode = "unsupported"
            supported = False

        diag_rows.append({
            "source_file": source_file,
            "scan": scan,
            "query_label": query_label,
            "rule_name": rule_name,
            "compendium": compendium,
            "section": section,
            "query_text": qtext,
            "diagnostic_mode": diagnostic_mode,
            "supported": supported,
        })

        # -------------------------------------------------
        # Mode 1: X + aminoaciddelta(...)
        # -------------------------------------------------
        if parsed_delta is not None:
            pairs = diagnose_delta_pairs_for_scan(mz, y_rel, qtext)

            for p in pairs:
                pair_rows.append({
                    "source_file": source_file,
                    "scan": scan,
                    "query_label": query_label,
                    "rule_name": rule_name,
                    "compendium": compendium,
                    "section": section,
                    **p,
                })

        # -------------------------------------------------
        # Mode 2: fixed MS2PROD targets
        # -------------------------------------------------
        elif fixed_targets is not None:
            tol = _extract_global_tolerancemz(qtext, default=0.02)
            min_int_pct = _extract_intensitypercent(qtext, default=0.0)

            for target in fixed_targets:
                idxs = np.where(np.abs(mz - float(target)) <= tol)[0]
                if len(idxs) == 0:
                    continue

                # keep only peaks above intensity threshold
                idxs = [idx for idx in idxs if y_rel[idx] >= min_int_pct]
                if len(idxs) == 0:
                    continue

                # choose the closest peak
                best = min(idxs, key=lambda idx: abs(float(mz[idx]) - float(target)))

                pair_rows.append({
                    "source_file": source_file,
                    "scan": scan,
                    "query_label": query_label,
                    "rule_name": rule_name,
                    "compendium": compendium,
                    "section": section,
                    "mode": "fixed_prod",
                    "prod_target": float(target),
                    "peak_mz": float(mz[best]),
                    "error_da": float(float(mz[best]) - float(target)),
                    "intensity_pct": float(y_rel[best]),
                    "tol": float(tol),
                })

    return pd.DataFrame(diag_rows), pd.DataFrame(pair_rows)
    
# ============================================================
# Streamlit UI
# ============================================================
st.title("MassQL Compendium Runner + MS/MS Viewer")
st.caption(
    "Upload an MS file (.mgf, .mzML, .mzXML) and one or more MassQL compendium .txt files. "
    "Tables render via HTML (no pyarrow). For uploaded MGF files, scan numbers are normalized to 1..N."
)

with st.sidebar:
    st.header("1) Inputs")

    up_ms_files = st.file_uploader(
        "MS file(s) (.mgf, .mzML, .mzXML)",
        type=["mgf", "mzml", "mzxml"],
        accept_multiple_files=True,
        help="Upload one or more MS files. For MGF, scans are forced to 1..N.",
    )

    up_comps = st.file_uploader(
        "Compendium files (txt/tsv) — supports QUERY-block and name<TAB>QUERY formats",
        type=["txt", "tsv", "tab", "csv"],
        accept_multiple_files=True,
        help="Either: (A) multi-line blocks starting with QUERY, or (B) TSV lines: name<TAB>QUERY ...",
    )

    preview_compendia = st.checkbox("Preview parsed compendia on upload", value=False)


    convert_ms1_mgf_flag = st.checkbox(
        "Force-convert uploaded MGF into simple mzML with all spectra written as MS1",
        value=False,
        help="If checked, any uploaded .mgf file will be converted to a simple MS1 mzML before running MassQL."
    )

    st.header("2) Qualifier overrides (applied to all)")
    colA, colB, colC = st.columns(3)
    with colA:
        tol_mz = st.text_input("TOLERANCEMZ override (blank = keep query value)", value="")
    with colB:
        tol_ppm = st.text_input("TOLERANCEPPM override (blank = keep query value)", value="")
    with colC:
        inten_pct = st.text_input("INTENSITYPERCENT override (blank = keep query value)", value="")

    st.markdown("You can also add **per-compendium** overrides by name below (JSON). Example:")
    st.code(
        '''{
  "*": {"TOLERANCEMZ": 0.03, "INTENSITYPERCENT": 10},
  "Flavonoids": {"TOLERANCEMZ": 0.02}
}''',
        language="json",
    )
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
def _persist_upload(uploaded):
    if not uploaded:
        return None, None
    tmpdir = tempfile.mkdtemp(prefix="massql_")
    orig = Path(uploaded.name).name if getattr(uploaded, "name", None) else "uploaded.dat"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", orig) or "uploaded.dat"
    outp = Path(tmpdir) / safe
    with open(outp, "wb") as f:
        f.write(uploaded.getbuffer())
    return str(outp), safe

state = st.session_state
if "combined" not in state:
    state.combined = pd.DataFrame()
    state.presence_by_comp = {}
    state.presence_global = pd.DataFrame()
    state.ms_paths = []
    state.ms_indexes = {}
    state.source_name_map = {}
    state.display_to_view_path = {}
    state.last_comp_paths = []

if "query_registry" not in state:
    state.query_registry = pd.DataFrame()
    
if "diagnostics_df" not in state:
    state.diagnostics_df = pd.DataFrame()

if "diagnostics_pairs" not in state:
    state.diagnostics_pairs = pd.DataFrame()

def _to_float_or_none(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _build_qualifier_overrides(global_inputs: dict, per_comp_text: str) -> dict | None:
    """
    Returns dict like:
    {
      "*": {"TOLERANCEMZ": 0.02, "TOLERANCEPPM": 10, "INTENSITYPERCENT": 5},
      "Flavonoids": {"TOLERANCEMZ": 0.01}
    }
    """
    base = {k: v for k, v in global_inputs.items() if v is not None}
    out = {"*": base} if base else {}

    per_comp_text = (per_comp_text or "").strip()
    if per_comp_text:
        try:
            obj = json.loads(per_comp_text)
            if isinstance(obj, dict):
                # shallow-merge user json (expects already correct structure)
                for k, v in obj.items():
                    if isinstance(v, dict):
                        out[k] = {kk: vv for kk, vv in v.items() if vv is not None}
        except Exception as e:
            st.warning(f"Overrides JSON could not be parsed. Ignoring. Error: {e}")

    return out if out else None


if run_btn:
    if (not up_ms_files) or (not up_comps):
        st.error("Please upload at least one MS file and one compendium file.")
        st.stop()

    # -----------------------------
    # Qualifier overrides
    # -----------------------------
    global_overrides = {
        "TOLERANCEMZ": _to_float_or_none(tol_mz),
        "TOLERANCEPPM": _to_float_or_none(tol_ppm),
        "INTENSITYPERCENT": _to_float_or_none(inten_pct),
    }
    qualifier_overrides = _build_qualifier_overrides(global_overrides, per_comp_json)

    # -----------------------------
    # Persist MS uploads
    # -----------------------------
    ms_paths_query: list[str] = []      # files used by MassQL
    ms_paths_view: list[str] = []       # original files used for viewer
    source_name_map_query: dict[str, str] = {}
    source_name_map_view: dict[str, str] = {}
    display_to_view_path: dict[str, str] = {}

    for f in up_ms_files:
        p, n = _persist_upload(f)
        if p is None:
            continue

        ms_paths_view.append(p)
        source_name_map_view[p] = n
        display_to_view_path[n] = p

        # default query path = original path
        qpath = p

        # optional conversion only for query engine
        if convert_ms1_mgf_flag and Path(p).suffix.lower() == ".mgf":
            qpath = convert_ms1_mgf_to_mzml(p)

        ms_paths_query.append(qpath)
        source_name_map_query[qpath] = n  # keep display name identical to original

    # -----------------------------
    # Persist + parse compendiums
    # -----------------------------
    comp_paths: list[str] = []
    parsed_compendia: list[dict] = []  # [{"comp_name": str, "qitems": [...], "warnings": [...]}]

    for f in up_comps:
        comp_p, comp_disp = _persist_upload(f)
        if comp_p is None:
            continue

        comp_paths.append(comp_p)

        raw = Path(comp_p).read_text(encoding="utf-8", errors="replace")
        comp_name = Path(comp_disp).stem

        qitems, warns = parse_compendium_auto(raw, fallback_name=comp_name)
        parsed_compendia.append(
            {
                "comp_name": comp_name,
                "qitems": qitems,
                "warnings": warns,
            }
        )

    # -----------------------------
    # Optional preview
    # -----------------------------
    if preview_compendia:
        preview_rows = []
        for pack in parsed_compendia:
            preview_rows.append({
                "compendium": pack["comp_name"],
                "queries_parsed": len(pack["qitems"]),
                "warnings_count": len(pack.get("warnings", [])),
                "warnings_preview": " | ".join(pack.get("warnings", [])[:3]),
            })
        prev_df = pd.DataFrame(preview_rows)
        st.markdown(html_table(prev_df, 100), unsafe_allow_html=True)

    # -----------------------------
    # Run MassQL
    # -----------------------------
    with st.spinner("Running MassQL compendiums..."):
        combined_unique, presence_by_comp, presence_global, query_registry = run_compendiums(
            compendium_files=comp_paths,
            ms_files=ms_paths_query,
            parsed_compendia=[
                {"comp_name": p["comp_name"], "qitems": p["qitems"]}
                for p in parsed_compendia
            ],
            use_loader_frames=False,
            parallel=False,
            qualifier_overrides=qualifier_overrides,
            source_name_map=source_name_map_query,
        )

    # -----------------------------
    # Build MS indexes for viewer
    # -----------------------------
    ms_indexes = {}
    for p in ms_paths_view:
        try:
            ms_indexes[p] = _build_ms_scan_index(p)
        except Exception as e:
            st.warning(f"[WARN] Could not index MS file {os.path.basename(p)}: {e}")
            ms_indexes[p] = {}

    # -----------------------------
    # Persist to session state
    # -----------------------------
    state.combined = combined_unique
    state.presence_by_comp = presence_by_comp
    state.presence_global = presence_global
    state.query_registry = query_registry

    state.ms_paths = ms_paths_view
    state.ms_indexes = ms_indexes
    state.source_name_map = source_name_map_view
    state.display_to_view_path = display_to_view_path
    state.last_comp_paths = comp_paths

    # add for diagnostics_check
    diagnostics_df, diagnostics_pairs = run_diagnostics_from_matches(
        combined_unique,
        ms_indexes,
        display_to_view_path,
    )

    state.diagnostics_df = diagnostics_df
    state.diagnostics_pairs = diagnostics_pairs

# ============================================================
# Results area
# ============================================================
combined = state.combined
presence_by_comp = state.presence_by_comp
presence_global = state.presence_global
ms_indexes = getattr(state, "ms_indexes", {})
source_name_map = getattr(state, "source_name_map", {})


st.markdown("---")

with st.expander("Queries executed (even with zero matches)", expanded=False):
    qr = getattr(state, "query_registry", pd.DataFrame())

    if qr is None or qr.empty:
        st.info("No query registry available yet. Run the compendiums first.")
    else:
        qrf = qr.copy()

        # -----------------------------
        # Filters
        # -----------------------------
        col_src, col_comp, col_status = st.columns(3)

        with col_src:
            src_opts = ["(all)"] + sorted(qrf["source_file"].fillna("").astype(str).unique().tolist())
            src_pick = st.selectbox("Source file", src_opts, index=0, key="qr_src")

        with col_comp:
            comp_opts = ["(all)"] + sorted(qrf["compendium"].fillna("").astype(str).unique().tolist())
            comp_pick = st.selectbox("Compendium", comp_opts, index=0, key="qr_comp")

        with col_status:
            qrf["status"] = qrf["status"].where(qrf["status"].notna(), "ok")
            status_opts = ["(all)"] + sorted(qrf["status"].astype(str).unique().tolist())
            status_pick = st.selectbox("Status", status_opts, index=0, key="qr_status")

        if src_pick != "(all)":
            qrf = qrf[qrf["source_file"].fillna("").astype(str) == src_pick]

        if comp_pick != "(all)":
            qrf = qrf[qrf["compendium"].fillna("").astype(str) == comp_pick]

        if status_pick != "(all)":
            qrf = qrf[qrf["status"].fillna("ok").astype(str) == status_pick]

        # -----------------------------
        # Table + download
        # -----------------------------
        table_cols = [
            c for c in [
                "source_file",
                "compendium",
                "section",
                "rule_name",
                "query_idx",
                "query_label",
                "or_patch",
                "status",
                "n_hits",
                "error",
            ]
            if c in qrf.columns
        ]

        if table_cols:
            st.markdown(html_table(qrf[table_cols], 100), unsafe_allow_html=True)
        else:
            st.info("No displayable columns found in the query registry.")

        st.download_button(
            "Download query_registry.csv",
            data=download_csv_bytes(qrf),
            file_name="query_registry.csv",
            key="dl_query_registry",
        )

        # -----------------------------
        # Show exact query text
        # -----------------------------
        required_cols = {"source_file", "query_label", "query_text"}
        if not qrf.empty and required_cols.issubset(qrf.columns):
            query_view = (
                qrf[["source_file", "query_label", "query_text"]]
                .dropna(subset=["query_label", "query_text"])
                .drop_duplicates()
                .reset_index(drop=True)
            )

            if not query_view.empty:
                query_view["display_label"] = (
                    query_view["source_file"].fillna("").astype(str)
                    + " :: "
                    + query_view["query_label"].fillna("").astype(str)
                )

                selected_label = st.selectbox(
                    "Show query text",
                    query_view["display_label"].tolist(),
                    index=0,
                    key="qry_show",
                )

                selected_text = query_view.loc[
                    query_view["display_label"] == selected_label,
                    "query_text"
                ].iloc[0]

                st.code(selected_text, language="text")

                st.download_button(
                    "Download shown_query.txt",
                    data=selected_text.encode("utf-8"),
                    file_name=f"{_safe_col(str(selected_label))}.txt",
                    mime="text/plain",
                    key="dl_shown_query",
                )

st.write("combined shape:", None if combined is None else combined.shape)

if combined is None or combined.empty:
    st.info("Upload inputs in the sidebar and press **Run MassQL Compendiums**.")
else:
    tab_results, tab_viewer, tab_diag = st.tabs([
        "Results",
        "MS/MS Viewer",
        "Diagnostics for sequence queries",
    ])

    # =========================================================
    # TAB 1 — RESULTS
    # =========================================================
    with tab_results:
        # -------------------------
        # Summary
        # -------------------------
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total result rows", f"{len(combined):,}")
        with c2:
            st.metric(
                "Unique scans (hits)",
                f"{combined['scan'].nunique() if 'scan' in combined.columns else 0:,}",
            )
        with c3:
            st.metric(
                "Compendiums",
                f"{combined['compendium'].nunique() if 'compendium' in combined.columns else 0:,}",
            )
        with c4:
            st.metric(
                "Sections",
                f"{combined['section'].nunique() if 'section' in combined.columns else 0:,}",
            )

        # -------------------------
        # Presence matrices
        # -------------------------
        st.markdown("---")
        st.markdown("### Global presence (compendium × section)")
        if presence_global is not None and not presence_global.empty:
            pg = presence_global.copy()
            pg.columns = [f"{c[0]} :: {c[1]}" for c in pg.columns.to_list()]
            pg = pg.reset_index()

            st.markdown(html_table(pg, 100), unsafe_allow_html=True)
            st.download_button(
                "Download presence_global.csv",
                data=download_csv_bytes(pg),
                file_name="presence_global.csv",
                key="dl_presence_global_tab",
            )
        else:
            st.info("No global presence matrix.")

        st.markdown("---")
        st.markdown("### Presence by compendium")
        if presence_by_comp:
            for comp, pres in presence_by_comp.items():
                st.markdown(f"**{comp}**")
                pres2 = pres.reset_index()
                st.markdown(html_table(pres2, 150), unsafe_allow_html=True)
                st.download_button(
                    f"Download presence_{_safe_col(comp)}.csv",
                    data=download_csv_bytes(pres2),
                    file_name=f"presence_{_safe_col(comp)}.csv",
                    key=f"dl_presence_{_safe_col(comp)}_tab",
                )
        else:
            st.info("No per-compendium presence matrices.")

        st.markdown("---")
        st.markdown("### Named presence table (compendium and section labels in cells)")
        named = make_named_presence_table(combined)
        st.markdown(html_table(named, 100), unsafe_allow_html=True)
        st.download_button(
            "Download combined_wide.csv",
            data=download_csv_bytes(named),
            file_name="combined_wide.csv",
            key="dl_combined_wide_tab",
        )

        st.markdown("---")
        st.markdown("### Coverage by compendium")
        cov = coverage_by_compendium(combined)
        st.markdown(html_table(cov, 200), unsafe_allow_html=True)
        st.download_button(
            "Download coverage.csv",
            data=download_csv_bytes(cov),
            file_name="coverage.csv",
            key="dl_coverage_tab",
        )

    # =========================================================
    # TAB 2 — MS/MS VIEWER
    # =========================================================
    with tab_viewer:
        st.subheader("Interactive MS/MS viewer (mpld3)")

        if "source_file" not in combined.columns:
            st.warning("Column 'source_file' is missing from results; cannot build MS/MS viewer.")
        else:
            sources = sorted(combined["source_file"].astype(str).dropna().unique().tolist())

            if not sources:
                st.info("No sources available for MS/MS viewer.")
            else:
                col_src, col_opts = st.columns([2, 3])

                with col_src:
                    chosen_source = st.selectbox(
                        "Source file",
                        sources,
                        index=0,
                        key="viewer_source",
                    )

                display_to_view_path = getattr(state, "display_to_view_path", {})
                ms_path = display_to_view_path.get(chosen_source)
                mgf_index = ms_indexes.get(ms_path, {}) if ms_path else {}

                df_src = combined.loc[
                    combined["source_file"].astype(str) == str(chosen_source)
                ].copy()

                scans: list[int] = []
                if "scan" in df_src.columns:
                    scans = sorted(
                        map(int, pd.to_numeric(df_src["scan"], errors="coerce").dropna().unique())
                    )

                if not scans and mgf_index:
                    scans = sorted(map(int, mgf_index.keys()))

                if not scans:
                    st.info("No scans available to display for this file.")
                else:
                    with col_opts:
                        chosen_scan = st.selectbox(
                            "Scan",
                            scans,
                            index=0,
                            key="viewer_scan",
                        )
                        normalize = st.checkbox(
                            "Normalize to 100%",
                            value=True,
                            key="viewer_normalize",
                        )
                        topn = st.slider(
                            "Annotate top-N peaks",
                            min_value=0,
                            max_value=40,
                            value=12,
                            step=1,
                            key="viewer_topn",
                        )

                    st.markdown("**MassQL hits for selected scan**")

                    dfscan = df_src.copy()
                    if "scan" in dfscan.columns:
                        dfscan = dfscan[
                            pd.to_numeric(dfscan["scan"], errors="coerce") == int(chosen_scan)
                        ]

                    if mgf_index:
                        html = _plot_ms2_html_from_index(
                            mgf_index,
                            int(chosen_scan),
                            normalize=normalize,
                            annotate_top_n=topn,
                            matched_queries_df=dfscan,
                        )
                        components.html(html, height=520, scrolling=True)
                    else:
                        st.warning("MS index not available to render spectrum for this file.")

                    keep_base = [
                        "source_file",
                        "scan",
                        "precmz",
                        "rt",
                        "compendium",
                        "section",
                        "rule_name",
                        "query_idx",
                        "query_label",
                    ]

                    keep_extra = [c for c in ("NAME", "SPECTRUMID") if c in dfscan.columns]
                    keep = [c for c in keep_base + keep_extra if c in dfscan.columns]

                    if not dfscan.empty and keep:
                        dfscan_view = (
                            dfscan[keep]
                            .drop_duplicates()
                            .sort_values(["source_file", "compendium", "section", "query_idx"])
                        )
                    else:
                        dfscan_view = dfscan

                    if not dfscan.empty and {"query_label", "query_text"}.issubset(dfscan.columns):
                        q_cols = [c for c in ["rule_name", "query_label", "query_text"] if c in dfscan.columns]

                        q_opts = (
                            dfscan[q_cols]
                            .dropna(subset=["query_text"])
                            .drop_duplicates()
                            .reset_index(drop=True)
                        )

                        if not q_opts.empty:
                            st.markdown("### Applied MassQL queries for this scan")

                            if "rule_name" in q_opts.columns:
                                q_opts["display_name"] = q_opts["rule_name"].fillna("").astype(str)
                                empty_mask = q_opts["display_name"].str.strip() == ""
                                q_opts.loc[empty_mask, "display_name"] = (
                                    q_opts.loc[empty_mask, "query_label"].fillna("").astype(str)
                                )
                            else:
                                q_opts["display_name"] = q_opts["query_label"].fillna("").astype(str)

                            chosen_q = st.selectbox(
                                "Select query",
                                q_opts["display_name"].tolist(),
                                index=0,
                                key=f"qdrop_{_safe_col(chosen_source)}_{chosen_scan}",
                            )

                            qtext_show = q_opts.loc[
                                q_opts["display_name"] == chosen_q,
                                "query_text"
                            ].iloc[0]

                            st.code(qtext_show, language="text")

                            st.download_button(
                                "Download selected_query.txt",
                                data=qtext_show.encode("utf-8"),
                                file_name=f"{_safe_col(chosen_source)}_scan_{chosen_scan}_{_safe_col(chosen_q)}.txt",
                                mime="text/plain",
                                key=f"dlq_{_safe_col(chosen_source)}_{chosen_scan}_{_safe_col(chosen_q)}",
                            )

                    st.markdown(html_table(dfscan_view, 100), unsafe_allow_html=True)
                    st.download_button(
                        "Download scan_hits.csv",
                        data=download_csv_bytes(dfscan_view),
                        file_name=f"{_safe_col(chosen_source)}_scan_{chosen_scan}_hits.csv",
                        key=f"dl_scan_hits_{_safe_col(chosen_source)}_{chosen_scan}",
                    )

    # =========================================================
    # TAB 3 — DIAGNOSTICS
    # =========================================================
    with tab_diag:
        st.subheader("Diagnostics from matched scans + matched queries")

        diagnostics_df = getattr(state, "diagnostics_df", pd.DataFrame())
        diagnostics_pairs = getattr(state, "diagnostics_pairs", pd.DataFrame())

        if diagnostics_df is None or diagnostics_df.empty:
            st.info("No diagnostics available yet. Run the compendiums first.")
        else:
            st.markdown("#### Query/scan pairs selected for diagnostics")
            st.markdown(html_table(diagnostics_df, 100), unsafe_allow_html=True)

            st.download_button(
                "Download diagnostics_scans.csv",
                data=download_csv_bytes(diagnostics_df),
                file_name="diagnostics_scans.csv",
                key="dl_diagnostics_scans",
            )

            if diagnostics_pairs is None or diagnostics_pairs.empty:
                st.warning("No diagnostic peaks were found for the supported query types.")
            else:
                st.markdown("#### Matched diagnostic rows")
                st.markdown(html_table(diagnostics_pairs, 200), unsafe_allow_html=True)

                st.download_button(
                    "Download diagnostics_pairs.csv",
                    data=download_csv_bytes(diagnostics_pairs),
                    file_name="diagnostics_pairs.csv",
                    key="dl_diagnostics_pairs",
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    ds_source = st.selectbox(
                        "Diagnostic source",
                        sorted(diagnostics_pairs["source_file"].astype(str).unique().tolist()),
                        key="diag_source",
                    )

                df1 = diagnostics_pairs[
                    diagnostics_pairs["source_file"].astype(str) == ds_source
                ].copy()

                with col2:
                    ds_scan = st.selectbox(
                        "Diagnostic scan",
                        sorted(df1["scan"].astype(int).unique().tolist()),
                        key="diag_scan",
                    )

                df2 = df1[df1["scan"].astype(int) == int(ds_scan)].copy()

                with col3:
                    ds_query = st.selectbox(
                        "Diagnostic query",
                        df2["query_label"].astype(str).drop_duplicates().tolist(),
                        key="diag_query",
                    )

                df3 = df2[df2["query_label"].astype(str) == ds_query].copy()

                sort_col = "error_da" if "error_da" in df3.columns else None
                if sort_col is not None:
                    df3_show = df3.sort_values(sort_col, key=lambda s: s.abs()).reset_index(drop=True).copy()
                else:
                    df3_show = df3.reset_index(drop=True).copy()

                st.markdown("#### Diagnostic rows for selected matched query")
                st.markdown(
                    html_table(df3_show, 100),
                    unsafe_allow_html=True,
                )

                if not df3_show.empty:
                    st.markdown("#### Diagnostic spectrum plot for selected row")

                    df3_show = df3_show.reset_index(drop=True)
                    df3_show["row_id"] = df3_show.index.astype(int)

                    diag_pick = st.selectbox(
                        "Select diagnostic row",
                        df3_show["row_id"].tolist(),
                        index=0,
                        key="diag_row_pick",
                    )

                    diag_row = df3_show.loc[df3_show["row_id"] == diag_pick].iloc[0].to_dict()

                    display_to_view_path = getattr(state, "display_to_view_path", {})
                    ms_path = display_to_view_path.get(ds_source)
                    mgf_index = ms_indexes.get(ms_path, {}) if ms_path else {}

                    diag_normalize = st.checkbox(
                        "Normalize diagnostic plot to 100%",
                        value=True,
                        key="diag_plot_normalize",
                    )

                    diag_topn = st.slider(
                        "Annotate top-N peaks in diagnostic plot",
                        min_value=0,
                        max_value=40,
                        value=12,
                        step=1,
                        key="diag_plot_topn",
                    )

                    if mgf_index:
                        diag_html = _plot_ms2_diagnostic_html(
                            mgf_index,
                            int(ds_scan),
                            diag_row,
                            normalize=diag_normalize,
                            annotate_top_n=diag_topn,
                        )
                        components.html(diag_html, height=560, scrolling=True)

                        html_filename = (
                            f"diagnostic_{_safe_col(ds_source)}"
                            f"_scan_{int(ds_scan)}"
                            f"_row_{int(diag_pick)}.html"
                        )

                        st.download_button(
                            "Download this diagnostic plot as HTML",
                            data=diag_html.encode("utf-8"),
                            file_name=html_filename,
                            mime="text/html",
                            key="dl_diag_single_html",
                        )

                        st.markdown("#### Batch export diagnostic plots")

                        if st.button("Generate HTML files for all rows in selected table", key="diag_batch_make"):
                            export_dir = Path(tempfile.mkdtemp(prefix="diag_htmls_"))
                            made_files = []

                            for ridx, rowx in df3_show.iterrows():
                                row_dict = rowx.to_dict()

                                html_text = _plot_ms2_diagnostic_html(
                                    mgf_index,
                                    int(ds_scan),
                                    row_dict,
                                    normalize=diag_normalize,
                                    annotate_top_n=diag_topn,
                                )

                                out_name = (
                                    f"diagnostic_{_safe_col(ds_source)}"
                                    f"_scan_{int(ds_scan)}"
                                    f"_row_{int(ridx)}.html"
                                )
                                out_path = export_dir / out_name
                                _save_html_text(html_text, str(out_path))
                                made_files.append(str(out_path))

                            if made_files:
                                zip_path = export_dir / "diagnostic_plots.zip"

                                import zipfile
                                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                    for fp in made_files:
                                        zf.write(fp, arcname=Path(fp).name)

                                st.success(f"Generated {len(made_files)} diagnostic HTML plot(s).")

                                with open(zip_path, "rb") as fzip:
                                    st.download_button(
                                        "Download all diagnostic plots (.zip)",
                                        data=fzip.read(),
                                        file_name="diagnostic_plots.zip",
                                        mime="application/zip",
                                        key="dl_diag_zip",
                                    )
                    else:
                        st.warning("MS index not available for diagnostic plotting.")