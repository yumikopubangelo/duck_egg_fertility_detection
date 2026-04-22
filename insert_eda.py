"""
Insert EDA section (3.2.3) before Praproses Data, then renumber 3.2.3–3.2.8 → 3.2.4–3.2.9.
Uses exact same tcPr element order and column widths as the existing tables in the document.
"""
from lxml import etree

DOC = "/tmp/gabungan/word/document.xml"
W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

def wp(tag): return f"{{{W}}}{tag}"

def make_para(text, bold=False, heading3=False,
              space_before="0", space_after="120"):
    p = etree.Element(wp("p"))
    pPr = etree.SubElement(p, wp("pPr"))
    if heading3:
        ps = etree.SubElement(pPr, wp("pStyle")); ps.set(wp("val"), "Heading3")
    sp = etree.SubElement(pPr, wp("spacing"))
    sp.set(wp("before"), space_before); sp.set(wp("after"), space_after)
    if heading3:
        rPr_pPr = etree.SubElement(pPr, wp("rPr"))
        c = etree.SubElement(rPr_pPr, wp("color")); c.set(wp("val"), "auto")
    r = etree.SubElement(p, wp("r"))
    rPr = etree.SubElement(r, wp("rPr"))
    rf = etree.SubElement(rPr, wp("rFonts"))
    rf.set(wp("ascii"), "Times New Roman")
    rf.set(wp("hAnsi"), "Times New Roman")
    rf.set(wp("cs"),    "Times New Roman")
    if bold or heading3:
        etree.SubElement(rPr, wp("b"))
        etree.SubElement(rPr, wp("bCs"))
    if heading3:
        c2 = etree.SubElement(rPr, wp("color")); c2.set(wp("val"), "auto")
    sz = etree.SubElement(rPr, wp("sz"));   sz.set(wp("val"), "24")
    sc = etree.SubElement(rPr, wp("szCs")); sc.set(wp("val"), "24")
    t = etree.SubElement(r, wp("t"))
    t.set(XML_SPACE, "preserve")
    t.text = text
    return p

def make_cell(text, col_w, is_header=False, is_even_row=False):
    """
    Exact tcPr order matching existing document tables:
    tcW -> tcBorders -> shd -> tcMar -> vAlign
    """
    header_fill = "1A3A6E"
    even_fill   = "EAF4FB"
    odd_fill    = "FFFFFF"
    border_color = header_fill if is_header else "BBBBBB"
    fill_color   = header_fill if is_header else (even_fill if is_even_row else odd_fill)
    text_color   = "FFFFFF" if is_header else None

    tc = etree.Element(wp("tc"))
    tcPr = etree.SubElement(tc, wp("tcPr"))

    # 1. tcW
    tcW = etree.SubElement(tcPr, wp("tcW"))
    tcW.set(wp("w"), str(col_w)); tcW.set(wp("type"), "dxa")

    # 2. tcBorders
    tcBdr = etree.SubElement(tcPr, wp("tcBorders"))
    for side in ("top", "left", "bottom", "right"):
        b = etree.SubElement(tcBdr, wp(side))
        b.set(wp("val"), "single"); b.set(wp("sz"), "4")
        b.set(wp("space"), "0");   b.set(wp("color"), border_color)

    # 3. shd
    shd = etree.SubElement(tcPr, wp("shd"))
    shd.set(wp("val"), "clear"); shd.set(wp("color"), "auto")
    shd.set(wp("fill"), fill_color)

    # 4. tcMar
    tcMar = etree.SubElement(tcPr, wp("tcMar"))
    for side, val in [("top","80"),("left","120"),("bottom","80"),("right","120")]:
        m = etree.SubElement(tcMar, wp(side))
        m.set(wp("w"), val); m.set(wp("type"), "dxa")

    # 5. vAlign
    va = etree.SubElement(tcPr, wp("vAlign")); va.set(wp("val"), "center")

    # paragraph inside cell
    p = etree.SubElement(tc, wp("p"))
    pPr = etree.SubElement(p, wp("pPr"))
    sp = etree.SubElement(pPr, wp("spacing"))
    sp.set(wp("before"), "0"); sp.set(wp("after"), "0")

    r = etree.SubElement(p, wp("r"))
    rPr = etree.SubElement(r, wp("rPr"))
    rf = etree.SubElement(rPr, wp("rFonts"))
    rf.set(wp("ascii"), "Times New Roman")
    rf.set(wp("hAnsi"), "Times New Roman")
    rf.set(wp("cs"),    "Times New Roman")
    if is_header:
        etree.SubElement(rPr, wp("b"))
        etree.SubElement(rPr, wp("bCs"))
    if text_color:
        c = etree.SubElement(rPr, wp("color")); c.set(wp("val"), text_color)
    s = etree.SubElement(rPr, wp("sz"));   s.set(wp("val"), "24")
    sc= etree.SubElement(rPr, wp("szCs")); sc.set(wp("val"), "24")
    t = etree.SubElement(r, wp("t"))
    t.set(XML_SPACE, "preserve")
    t.text = text
    return tc

def make_table(rows_data):
    """
    Two columns: 2708 + 6318 = 9026 DXA — matches existing tables.
    rows_data = list of (col1, col2)
    """
    COL1, COL2 = 2708, 6318
    tbl = etree.Element(wp("tbl"))
    tblPr = etree.SubElement(tbl, wp("tblPr"))
    tblW = etree.SubElement(tblPr, wp("tblW"))
    tblW.set(wp("w"), "9026"); tblW.set(wp("type"), "dxa")
    tblGrid = etree.SubElement(tbl, wp("tblGrid"))
    for w in [COL1, COL2]:
        gc = etree.SubElement(tblGrid, wp("gridCol")); gc.set(wp("w"), str(w))

    for i, (c1, c2) in enumerate(rows_data):
        tr = etree.SubElement(tbl, wp("tr"))
        is_header = (i == 0)
        is_even   = (i % 2 == 0) and not is_header
        tr.append(make_cell(c1, COL1, is_header=is_header, is_even_row=is_even))
        tr.append(make_cell(c2, COL2, is_header=is_header, is_even_row=is_even))
    return tbl

# ─── EDA table data ──────────────────────────────────────────
EDA_ROWS = [
    ("Parameter / Aspek", "Standarisasi / Spesifikasi"),
    ("Distribusi Kelas",
     "Hitung jumlah citra fertil vs. infertil; deteksi class imbalance; tentukan kebutuhan oversampling atau augmentasi tambahan"),
    ("Analisis Intensitas Piksel",
     "Histogram intensitas per channel (R, G, B) dan grayscale; hitung mean, std, min, max per kelas untuk memahami karakteristik optik citra candling"),
    ("Justifikasi Parameter CLAHE",
     "Visualisasi histogram citra sebelum dan sesudah enhancement; tentukan clip_limit optimal (default = 2.0) berdasarkan distribusi intensitas aktual citra candling telur bebek"),
    ("Justifikasi Target Resize",
     "Analisis resolusi asli citra dan estimasi diameter rata-rata telur dalam piksel; konfirmasi bahwa 512 x 512 piksel memadai tanpa kehilangan detail embrio"),
    ("Penilaian Kualitas Citra",
     "Deteksi citra blur menggunakan variance of Laplacian (threshold < 100); flagging citra overexposed (mean > 220) dan underexposed (mean < 30) untuk dibuang sebelum training"),
    ("Visualisasi Sampel per Kelas",
     "Grid 5 x 5 citra per kelas (fertil dan infertil) untuk inspeksi visual dan konfirmasi konsistensi labeling"),
    ("Analisis Separabilitas Awal",
     "Scatter plot mean intensitas vs. label; PCA 2-D dari fitur dasar untuk estimasi apakah dua kelas dapat dipisahkan secara visual sebelum model dilatih"),
    ("Identifikasi Outlier",
     "Deteksi citra dengan intensitas ekstrem atau ukuran ROI di luar rentang normal menggunakan z-score (threshold > 3 std); dieksklusi dari dataset"),
    ("Output EDA",
     "Laporan statistik deskriptif, visualisasi distribusi, dan justifikasi tertulis untuk setiap keputusan parameter praproses"),
    ("Tool / Library",
     "Python: Matplotlib, Seaborn, OpenCV, NumPy, scikit-learn (PCA)"),
]

# ─── Build EDA content block ────────────────────────────────
eda_heading = make_para(
    "3.2.3 Analisis Eksplorasi Dataset (EDA)",
    heading3=True, space_before="240", space_after="120"
)
eda_desc = make_para(
    "Analisis eksplorasi dataset dilakukan untuk memahami karakteristik statistik dan visual "
    "citra candling telur bebek sebelum masuk ke tahap praproses. Tahapan ini bertujuan "
    "memvalidasi kualitas data yang telah dikumpulkan, mendeteksi ketidakseimbangan kelas "
    "(class imbalance), serta menjustifikasi setiap nilai parameter yang digunakan pada tahap "
    "praproses berikutnya. Hasil EDA juga memberikan dasar empiris untuk menentukan kebutuhan "
    "augmentasi dan estimasi separabilitas dua kelas (fertil dan infertil) secara awal.",
    space_before="0", space_after="120"
)
eda_table = make_table(EDA_ROWS)
empty_p = make_para("", space_before="0", space_after="80")

new_block = [eda_heading, eda_desc, eda_table, empty_p]

# ─── Parse and modify document ──────────────────────────────
tree = etree.parse(DOC)
body = tree.find(f"{{{W}}}body")
children = list(body)

# Find 3.2.3 Praproses heading
praproses_idx = None
for i, el in enumerate(children):
    txt = "".join(t.text or "" for t in el.iter(wp("t")))
    if "3.2.3 Praproses" in txt:
        praproses_idx = i; break

print(f"Praproses heading at body index {praproses_idx}")

# Insert EDA block before Praproses heading
for j, el in enumerate(new_block):
    body.insert(praproses_idx + j, el)

# Renumber existing sections (do in reverse order to avoid conflicts)
renumber_map = [
    ("3.2.8 Pengujian", "3.2.9 Pengujian"),
    ("3.2.7 Evaluasi",  "3.2.8 Evaluasi"),
    ("3.2.6 Klasifikasi", "3.2.7 Klasifikasi"),
    ("3.2.5 Ekstraksi", "3.2.6 Ekstraksi"),
    ("3.2.4 Segmentasi", "3.2.5 Segmentasi"),
    ("3.2.3 Praproses",  "3.2.4 Praproses"),
]
for old_pfx, new_pfx in renumber_map:
    for t_el in body.iter(wp("t")):
        if (t_el.text or "").startswith(old_pfx):
            t_el.text = t_el.text.replace(old_pfx, new_pfx, 1)
            print(f"  {old_pfx!r} -> {new_pfx!r}")

# Write back
with open(DOC, "wb") as f:
    f.write(etree.tostring(tree, xml_declaration=True,
                           encoding="UTF-8", standalone=True))
print("Written OK")
