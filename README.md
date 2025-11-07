### **Short GitHub description**

> Streamlit-based MassQL Compendium Runner and MS/MS Viewer — execute MassQL queries from multiple compendiums against MGF files, visualize hits, and inspect annotated spectra interactively (built on the MassQL framework by Wang *et al.*, *Nat. Methods* 2025).

---

### **README.md (recommended content)**

````markdown
# MassQL Compendium Runner + MS/MS Viewer

Interactive Streamlit app for executing **MassQL** queries across one or more compendium `.txt` files against `.mgf` spectral data, visualizing matched MS/MS spectra, and exporting coverage tables — all **without pyarrow** dependencies.

---

## Background

This interface builds upon the **Mass Query Language (MassQL)** framework:

> Wang, M., et al. **"MassQL: A query language for mass spectrometry data."**  
> *Nature Methods* (2025). [https://doi.org/10.1038/s41592-025-02660-z](https://doi.org/10.1038/s41592-025-02660-z)

Official MassQL documentation: [https://mwang87.github.io/MassQueryLanguage_Documentation/](https://mwang87.github.io/MassQueryLanguage_Documentation/)

---

## Features

- Upload **one `.mgf` file** and **multiple MassQL compendiums** (`.txt`)
- Apply **global or per-compendium qualifier overrides**
- Generate:
  - Hit summary tables (`summary.csv`)
  - Global and per-compendium presence matrices
  - Named “wide” coverage tables
- View **interactive MS/MS spectra** (zoom + hover via `mpld3`)
- Export results as CSVs (HTML rendering only, no `pyarrow`)

---

## Requirements

```bash
pip install streamlit pyteomics massql pandas numpy matplotlib mpld3 pillow
````

---

## Usage

Run locally:

```bash
streamlit run app.py
```

Then open the app in your browser (default: `http://localhost:8501`).

---

## Input Format

* **MGF file:** Standard `.mgf` with `SCANS`, `PEPMASS`, and `RTINSECONDS` fields.
* **Compendium files:** `.txt` files containing one or more MassQL queries, separated by comment headers `# SectionName`.

Example:

```text
# Hexose
QUERY scaninfo(MS2DATA) WHERE MS2NL=162.05:TOLERANCEMZ=0.05:INTENSITYPERCENT=1

# Deoxyhexose
QUERY scaninfo(MS2DATA) WHERE MS2NL=146.05:TOLERANCEMZ=0.05:INTENSITYPERCENT=1
```

---

## Citation

If you use this app or derive results from MassQL, please cite:

> **MassQL: A query language for mass spectrometry data.**
> Wang, M., et al. *Nature Methods* 22, 123–129 (2025).
> DOI: [10.1038/s41592-025-02660-z](https://doi.org/10.1038/s41592-025-02660-z)

---

## Author and Lab

Developed by **Ricardo M. Borges**
Laboratório de Análise e Aplicação de Biomoléculas (LAABio) – IPPN/UFRJ
[https://github.com/RicardoMBorges](https://github.com/RicardoMBorges)

---

## License

This repository is distributed under the **MIT License**, with proper attribution to the original MassQL framework.

---

### Keywords

`MassQL` • `MS/MS` • `metabolomics` • `compendium` • `mass spectrometry` • `streamlit` • `pyteomics`

