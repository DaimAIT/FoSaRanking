# streamlit_app_optimized.py
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
from paddleocr import PaddleOCR
from io import BytesIO
import xlsxwriter
import gc

# ---------------- OCR ----------------
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# OCR исправления
fix_map = {
    "ђ": "d", "ј": "j", "リ": "y", "ユ": "U",
    "+": "t", "а": "a", "Р": "P", "џ": "p", "С": "C",
    "Sagao": "Saga", "Sagad": "Saga", "Forgotren": "Forgotten"
}

KNOWN_TAGS = {"FoSa": "Forgotten Saga"}

def fix_row(row):
    if row["Clan tag"] in KNOWN_TAGS:
        row["Clan"] = KNOWN_TAGS[row["Clan tag"]]

    if row["Clan"] == "N/A" and row["Clan tag"] not in KNOWN_TAGS and row["Clan tag"] != "N/A":
        row["Nickname"] = (row["Nickname"] + " " + row["Clan tag"]).strip()
        row["Clan tag"] = "N/A"
    return row

def clean_text(s: str) -> str:
    s = s.strip()
    for bad, good in fix_map.items():
        s = s.replace(bad, good)
    s = re.sub(r"\s+", " ", s)
    return s

# ---------------- Image Processing ----------------
def crop_table_body(img, top_ratio=0.22, bottom_ratio=0.06):
    h, w = img.shape[:2]
    top = int(h * top_ratio)
    bottom = int(h * (1 - bottom_ratio))
    return img[top:bottom, :]

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Лёгкая предобработка для ускорения
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced  # без бинаризации для сохранения деталей

def parse_boxes(ocr_res):
    out = []
    for b, (txt, conf) in ocr_res:
        coords = np.array(b)
        x, y = coords[:, 0].mean(), coords[:, 1].mean()
        out.append((x, y, txt.strip()))
    return out

def extract_rows_from_image(img):
    img = crop_table_body(img, 0.24, 0.20)
    img = preprocess_image(img)
    h, w = img.shape[:2]
    
    ocr_res = ocr.ocr(img, cls=True)[0]
    boxes = parse_boxes(ocr_res)

    left_bound = w * 0.20
    right_bound = w * 0.75

    left   = [(x, y, t) for x, y, t in boxes if x < left_bound]
    center = [(x, y, t) for x, y, t in boxes if left_bound <= x <= right_bound]
    right  = [(x, y, t) for x, y, t in boxes if x > right_bound]

    # Очки
    scores = []
    for x, y, t in right:
        nums = re.findall(r"\d[\d,\.]+", t)
        for num in nums:
            val = num.replace(",", "").replace(".", "")
            if val.isdigit():
                scores.append((y, int(val)))
    scores.sort(key=lambda x: x[0])

    # Игроки
    rows = []
    for sy, sc in scores:
        same_line = [c for c in center if abs(c[1] - sy) < h * 0.05]
        if not same_line:
            continue

        same_line.sort(key=lambda c: c[0])
        nick_box = next((t for x, y, t in same_line if not t.strip().startswith("[")), None)
        clan_box = next((t for x, y, t in same_line if t.strip().startswith("[")), None)

        nick = clean_text(nick_box) if nick_box else "N/A"
        clan_tag, clan_name = "FoSa", "Forgotten Saga"

        if clan_box:
            m = re.search(r"\[([^\]]+)\]\s*(.*)", clan_box)
            if m:
                clan_tag = m.group(1)
                clan_name = m.group(2).strip() or KNOWN_TAGS.get(clan_tag, "N/A")

        rows.append({
            "Nickname": nick,
            "Clan tag": clan_tag,
            "Clan": clan_name,
            "Points": sc
        })


    return rows

def extract_rows(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rows = extract_rows_from_image(img)
    del img, img_array
    gc.collect()
    return rows

# ---------------- Excel Export ----------------
def create_excel(df):
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet("Players")
    
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name)

    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        for col_idx, value in enumerate(row):
            worksheet.write(row_idx, col_idx, value)

    workbook.close()
    output.seek(0)
    return output


@st.cache_data
def process_uploaded_files(uploaded_files):
    all_rows = []
    for file in uploaded_files:
        content = file.read()
        rows = extract_rows(content)
        all_rows.extend(rows)
        del content, rows
        import gc; gc.collect()
    return all_rows

# ---------------- Streamlit Interface ----------------
st.set_page_config(page_title="OCR Table Parser", layout="wide")
st.title("📄 FoSa VS Points OCR Parser (Optimized)")

uploaded_files = st.file_uploader(
    "Upload images",  type=['jpg','png','jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    all_rows = process_uploaded_files(uploaded_files)
    all_rows = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        content = file.read()
        rows = extract_rows(content)
        all_rows.extend(rows)

        del content, rows
        gc.collect()

        progress_bar.progress((i + 1)/len(uploaded_files))

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.apply(fix_row, axis=1)
        df.sort_values("Points", inplace=True)
        df = df.drop_duplicates(subset=["Nickname"], keep="first")
        df.reset_index(drop=True, inplace=True)
        st.success("✅ All images processed!")

        st.dataframe(df)

        # Скачать Excel
        excel_output = create_excel(df)
        st.download_button(
            label="⬇️ Download Excel",
            data=excel_output,
            file_name="players_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Лёгкий CSV вариант
  #      csv_data = df.to_csv(index=False).encode('utf-8')
   #     st.download_button("⬇️ Download CSV", data=csv_data, file_name="players_final.csv", mime="text/csv")

        del df
        gc.collect()
