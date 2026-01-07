# app.py
import re
import json
import base64
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="DNA→RNA→Protein + Free 3D Viewer", layout="centered")
st.title("DNA → RNA → Protein + Free Structure Options")

# =========================================================
# Links (free routes)
# =========================================================
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"   # /<UniProtAcc>
ALPHAFOLD_ENTRY = "https://www.alphafold.ebi.ac.uk/entry"     # /<UniProtAcc>
COLABFOLD_NOTEBOOK = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"

AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

# =========================================================
# Helpers: sequence cleaning & conversion
# =========================================================
def clean_dna_keep_len(seq: str) -> str:
    """Remove FASTA headers; keep letters only; convert non-ATGC to N (keep length stable)."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def dna_to_rna(dna: str) -> str:
    """DNA -> RNA (T -> U)"""
    return dna.replace("T", "U")

def translate_from_dna(dna: str, strand: str, frame: int) -> str:
    """
    Translate from DNA using Biopython (includes '*' stops).
    strand: '+' or '-'
    frame: 0/1/2
    """
    seq = dna
    if strand == "-":
        seq = str(Seq(dna).reverse_complement())
    seq = seq[frame:]
    seq = seq[: (len(seq) // 3) * 3]
    return str(Seq(seq).translate(to_stop=False)) if seq else ""

def best_orf_6frames(dna: str):
    """Pick the longest AA segment before the first stop among 6 frames."""
    best = {"strand": "+", "frame": 0, "aa_full": "", "aa_orf": "", "orf_len": 0, "stop_count": 10**9}
    for strand in ["+", "-"]:
        for frame in [0, 1, 2]:
            aa_full = translate_from_dna(dna, strand, frame)
            if not aa_full:
                continue
            aa_orf = aa_full.split("*")[0]
            orf_len = len(aa_orf)
            stop_count = aa_full.count("*")
            if (orf_len > best["orf_len"]) or (orf_len == best["orf_len"] and stop_count < best["stop_count"]):
                best = {
                    "strand": strand,
                    "frame": frame,
                    "aa_full": aa_full,
                    "aa_orf": aa_orf,
                    "orf_len": orf_len,
                    "stop_count": stop_count,
                }
    return best

def sanitize_protein(aa: str) -> str:
    aa = (aa or "").upper()
    return "".join([c for c in aa if c in AA_VALID])

# =========================================================
# Helpers: UniProt normalization/validation (fix 400 Bad Request)
# =========================================================
def normalize_uniprot_id(raw: str) -> str:
    """
    Accept common inputs:
      - '>sp|P00520|...' -> P00520
      - 'tr|A0A0D8HKK8|..' -> A0A0D8HKK8
      - 'P00520-2' -> P00520 (strip isoform)
      - '  p00520  ' -> P00520
    """
    s = (raw or "").strip()
    s = s.replace(">", "").strip()

    # If FASTA-like with pipes: sp|ACC|NAME
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        # Usually accession is the 2nd token in sp|ACC|NAME
        for token in parts:
            t = token.strip().upper()
            if re.fullmatch(r"[A-Z0-9\-]+", t):
                # keep scanning; we will validate later
                s = t
                # don't break too early; but good enough
                break

    s = s.strip().upper()
    # strip isoform suffix like -2
    s = re.sub(r"-\d+$", "", s)
    # remove spaces again
    s = s.strip()
    return s

def is_probably_uniprot_accession(acc: str) -> bool:
    """
    Pragmatic validator to prevent most AlphaFold API 400s.
    Common UniProt accessions:
      - 6 chars (e.g., P12345, Q8ZIN0)
      - 10 chars (newer, e.g., A0A0B4J2D5)
    """
    if not acc:
        return False
    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", acc):
        return True
    if re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", acc):
        return True
    if re.fullmatch(r"[A-Z0-9]{10}", acc):
        return True
    return False

# =========================================================
# Helpers: QC-friendly visualization without matplotlib
# =========================================================
def kyte_doolittle(aa: str, window=19):
    kd = {
        "I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,
        "G":-0.4,"T":-0.7,"S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,
        "H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5
    }
    aa = sanitize_protein(aa)
    if len(aa) < window:
        return [], []
    pos, vals = [], []
    for i in range(len(aa) - window + 1):
        seg = aa[i:i+window]
        vals.append(sum(kd[x] for x in seg) / window)
        pos.append(i + 1)  # 1-based window start
    return pos, vals

# =========================================================
# Helpers: HTTP
# =========================================================
def http_get_json(url: str, timeout=60):
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "streamlit-qc-app/1.0",
        }
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

def http_get_bytes(url: str, timeout=120):
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={"User-Agent": "streamlit-qc-app/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read()

def alphafold_lookup(uniprot_acc: str):
    """GET AlphaFold DB prediction JSON for UniProt accession."""
    url = f"{ALPHAFOLD_API}/{uniprot_acc}"
    return http_get_json(url)

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
dna_raw = st.text_area("Paste DNA sequence (FASTA / plain)", height=220)

mode = st.radio("Translation mode", ["Auto (best ORF)", "Manual"], horizontal=True)
c1, c2 = st.columns(2)
strand = c1.selectbox("Strand", ["+", "-"], disabled=(mode == "Auto (best ORF)"))
frame = c2.selectbox("Frame", [0, 1, 2], disabled=(mode == "Auto (best ORF)"))

window = st.slider("Hydropathy window", 7, 31, 19, step=2)

st.markdown("---")
st.subheader("Free structure options")

with st.expander("Option A (FREE): ColabFold / AlphaFold2 (run outside)"):
    st.write(
        "ใช้ ColabFold (ฟรี) เพื่อทำนายโครงสร้างจาก **protein sequence** แล้วดาวน์โหลดไฟล์ **PDB / mmCIF** "
        "กลับมาอัปโหลดที่ Option C เพื่อดู 3D ในแอปนี้"
    )
    st.link_button("Open ColabFold (AlphaFold2 notebook)", COLABFOLD_NOTEBOOK)

with st.expander("Option B (FREE if exists): AlphaFold DB by UniProt accession"):
    uniprot_raw = st.text_input("UniProt Accession (e.g., P12345 / A0A0B4J2D5)", value="")
    st.caption("ต้องเป็น UniProt accession เท่านั้น (ไม่ใช่ชื่อ gene/protein)")

with st.expander("Option C: Upload a structure file (PDB/mmCIF) to view in 3D"):
    uploaded = st.file_uploader("Upload .pdb or .cif", type=["pdb", "cif"])

# =========================================================
# Main run
# =========================================================
if st.button("Convert & Analyze", type="primary"):
    DNA = clean_dna_keep_len(dna_raw)
    if not DNA:
        st.error("Invalid DNA sequence.")
        st.stop()

    RNA = dna_to_rna(DNA)

    # Translate (from DNA; RNA shown for reporting)
    if mode == "Auto (best ORF)":
        best = best_orf_6frames(DNA)
        strand_use = best["strand"]
        frame_use = best["frame"]
        aa_orf = best["aa_orf"]
        stop_count = best["stop_count"]
        note = f"Auto ORF: strand {strand_use}, frame {frame_use} | ORF={best['orf_len']} aa | stops(full)={stop_count}"
    else:
        strand_use = strand
        frame_use = int(frame)
        aa_full = translate_from_dna(DNA, strand_use, frame_use)
        aa_orf = aa_full.split("*")[0] if aa_full else ""
        stop_count = aa_full.count("*") if aa_full else 0
        note = f"Manual: strand {strand_use}, frame {frame_use} | ORF={len(aa_orf)} aa | stops(full)={stop_count}"

    protein = sanitize_protein(aa_orf)
    if not protein:
        st.error("Protein translation failed (check frame/strand or too many ambiguities).")
        st.stop()

    # Protein properties
    pa = ProteinAnalysis(protein)

    st.markdown("## ✅ Outputs")
    st.write(f"**Sample:** {sample_id or '-'}")
    st.write(f"**DNA length:** {len(DNA)} nt")
    st.write(f"**RNA length:** {len(RNA)} nt")
    st.write(f"**Translation:** {note}")
    st.write(f"**Protein length:** {len(protein)} aa")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MW (Da)", f"{pa.molecular_weight():,.1f}")
    m2.metric("pI", f"{pa.isoelectric_point():.2f}")
    m3.metric("GRAVY", f"{pa.gravy():.2f}")
    m4.metric("Aromaticity", f"{pa.aromaticity():.3f}")

    st.markdown("### DNA (cleaned)")
    st.code(DNA, language="text")

    st.markdown("### RNA (T→U)")
    st.code(RNA, language="text")

    st.markdown("### Protein (ORF before first stop)")
    fasta = f">{sample_id or 'protein'}|strand={strand_use}|frame={frame_use}\n{protein}\n"
    st.code(fasta, language="text")

    # Hydropathy plot (Streamlit native)
    st.markdown("## 📈 Protein visualization (in-app)")
    pos, vals = kyte_doolittle(protein, window=window)
    if vals:
        df = pd.DataFrame({"Hydropathy (Kyte-Doolittle)": vals}, index=pos)
        st.line_chart(df)
    else:
        st.info("Protein too short for this hydropathy window.")

    # Downloads
    st.markdown("## ⬇️ Downloads")
    st.download_button("Download DNA (txt)", DNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_DNA.txt")
    st.download_button("Download RNA (txt)", RNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_RNA.txt")
    st.download_button("Download Protein FASTA", fasta.encode("utf-8"), file_name=f"{sample_id or 'protein'}_protein.fasta")

    # =====================================================
    # 3D structure viewer (FREE routes)
    # =====================================================
    st.markdown("---")
    st.markdown("## 🧊 3D Structure Viewer")

    structure_bytes = None
    structure_ext = None
    structure_source = None

    # Option C: uploaded structure
    if uploaded is not None:
        structure_bytes = uploaded.getvalue()
        structure_ext = uploaded.name.split(".")[-1].lower()
        structure_source = f"Uploaded file: {uploaded.name}"

    # Option B: AlphaFold DB by UniProt
    elif (uniprot_raw or "").strip():
        acc = normalize_uniprot_id(uniprot_raw)

        if not is_probably_uniprot_accession(acc):
            st.error(
                "UniProt accession ดูไม่ถูกต้องครับ\n"
                "- ตัวอย่างที่ถูก: P12345, Q8ZIN0, A0A0B4J2D5\n"
                "- ถ้าเป็น sp|ACC|NAME ให้ใส่ ACC อย่างเดียวก็ได้"
            )
            st.stop()

        try:
            with st.spinner(f"Querying AlphaFold DB API for {acc} ..."):
                _, data = alphafold_lookup(acc)

            # AlphaFold API typically returns a list
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                entry = data[0]
                pdb_url = entry.get("pdbUrl") or entry.get("pdb_url")
                cif_url = entry.get("cifUrl") or entry.get("cif_url")

                if pdb_url:
                    with st.spinner("Downloading AlphaFold PDB..."):
                        _, structure_bytes = http_get_bytes(pdb_url)
                    structure_ext = "pdb"
                    structure_source = f"AlphaFold DB: {acc} (PDB)"
                elif cif_url:
                    with st.spinner("Downloading AlphaFold mmCIF..."):
                        _, structure_bytes = http_get_bytes(cif_url)
                    structure_ext = "cif"
                    structure_source = f"AlphaFold DB: {acc} (mmCIF)"
                else:
                    st.warning("Found AlphaFold entry but no pdbUrl/cifUrl was provided.")
                    st.code(json.dumps(entry, indent=2))
                    st.stop()
            else:
                st.warning("No AlphaFold prediction found (or unexpected response).")
                st.link_button("Open AlphaFold entry page", f"{ALPHAFOLD_ENTRY}/{acc}")
                st.stop()

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            st.error(f"AlphaFold API HTTPError: {e.code} {e.reason}")
            st.write("Debug (server message):")
            st.code(body)
            st.stop()
        except Exception as ex:
            st.error(f"AlphaFold API Error: {ex}")
            st.stop()

    # If still none, guide user
    if not structure_bytes:
        st.info(
            "ยังไม่มีโครงสร้างให้แสดง 3D:\n"
            "- ใช้ Option A (ColabFold) เพื่อสร้างโครงสร้างฟรี แล้วอัปโหลดไฟล์ PDB/mmCIF ใน Option C\n"
            "- หรือใช้ Option B ใส่ UniProt accession (ถ้ามีอยู่ใน AlphaFold DB)\n"
        )
        st.stop()

    st.success(f"Structure ready ✅ ({structure_source})")

    # Render 3D with NGL (embedded)
    # NGL loads PDB/mmCIF in the browser.
    ext = structure_ext if structure_ext in ["pdb", "cif"] else "pdb"
    b64 = base64.b64encode(structure_bytes).decode("utf-8")

    ngl_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>
  <style>
    body {{ margin:0; }}
    #viewport {{ width: 100%; height: 540px; }}
  </style>
</head>
<body>
  <div id="viewport"></div>
  <script>
    const stage = new NGL.Stage("viewport");
    window.addEventListener("resize", function(){{ stage.handleResize(); }}, false);

    const b64 = "{b64}";
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i=0; i<len; i++) bytes[i] = binary.charCodeAt(i);

    const blob = new Blob([bytes], {{type: "text/plain"}});
    stage.loadFile(blob, {{ ext: "{ext}" }}).then(function(o){{
      o.addRepresentation("cartoon");
      o.autoView();
    }});
  </script>
</body>
</html>
"""
    st.components.v1.html(ngl_html, height=560, scrolling=False)

    st.download_button(
        "Download structure file",
        data=structure_bytes,
        file_name=f"{sample_id or 'structure'}.{ext}",
        mime="chemical/x-pdb" if ext == "pdb" else "chemical/x-cif",
    )

st.caption(
    "Tip: ถ้า AlphaFold API ขึ้น 400 ให้ตรวจว่าใส่ UniProt accession ถูกต้อง (ไม่ใช่ชื่อ gene/protein) "
    "และไม่มีตัวอักษร/ช่องว่างแปลก ๆ. สำหรับ sequence ใหม่ แนะนำใช้ ColabFold แล้วอัปโหลด PDB/mmCIF กลับมาดูในแอป."
)
