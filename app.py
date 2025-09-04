# streamlit_app.py
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
from paddleocr import PaddleOCR
from io import BytesIO

# OCR движок
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# Фикс OCR ошибок
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

def crop_table_body(img, top_ratio=0.22, bottom_ratio=0.06):
    h, w = img.shape[:2]
    top = int(h * top_ratio)
    bottom = int(h * (1 - bottom_ratio))
    return img[top:bottom, :]

def parse_boxes(ocr_res):
    out = []
    for b, (txt, conf) in ocr_res:
        x = sum([p[0] for p in b]) / 4
        y = sum([p[1] for p in b]) / 4
        out.append((x, y, txt.strip()))
    return out

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    return denoised  # без бинаризации

def extract_rows_from_image(img):
    img = crop_table_body(img, 0.22, 0.08)
    img = preprocess_image(img)
    h, w = img.shape[:2]

    ocr_res = ocr.ocr(img, cls=True)[0]
    boxes = parse_boxes(ocr_res)

    # Центр ников = 20–85% ширины
    left_bound = w * 0.20
    right_bound = w * 0.85

    left   = [(x, y, t) for x, y, t in boxes if x < left_bound]
    center = [(x, y, t) for x, y, t in boxes if left_bound <= x <= right_bound]
    right  = [(x, y, t) for x, y, t in boxes if x > right_bound]

    # Ранги
    ranks = [(y, int(re.sub(r"\D", "", t))) for x,y,t in left if re.sub(r"\D","",t).isdigit()]
    ranks.sort(key=lambda x: x[0])

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
    for i, (sy, sc) in enumerate(scores):
        rank = next((r for ry, r in ranks if abs(ry-sy) < h*0.04), i+1)
        closest_txt, min_dy = None, 999999
        for cx, cy, ct in center:
            dy = abs(cy-sy)
            if dy < min_dy:
                closest_txt, min_dy = ct, dy
        if not closest_txt:
            continue

        mid_line = clean_text(closest_txt)
        nick, clan_tag, clan_name = mid_line, "FoSa", "Forgotten Saga"
        m = re.search(r"\[([^\]]+)\]", mid_line)
        if m:
            clan_tag = m.group(1)
            clan_name = mid_line[m.end():].strip()
            nick = mid_line[:m.start()].strip()
        else:
            parts = mid_line.split()
            if len(parts) > 1:
                nick = " ".join(parts[:-1])
                clan_tag = parts[-1]

        rows.append({
            "Rank": rank,
            "Nickname": nick,
            "Clan tag": clan_tag,
            "Clan": clan_name,
            "Points": sc
        })

    return rows


def extract_rows(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return extract_rows_from_image(img)
# ---------------- Streamlit интерфейс ----------------
st.set_page_config(page_title="OCR Table Parser", layout="wide")
st.title("📄 FoSa VS Points OCR Parser")

uploaded_files = st.file_uploader("Upload images ", accept_multiple_files=True, type=['jpg','png','jpeg'])

if uploaded_files:
    all_rows = []
    progress_bar = st.progress(0)
    for i, file in enumerate(uploaded_files):
        content = file.read()
        rows = extract_rows(content)
        all_rows.extend(rows)
        progress_bar.progress((i + 1)/len(uploaded_files))

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.apply(fix_row, axis=1)
        df.sort_values("Rank", inplace=True)
        df = df.drop_duplicates(subset=["Nickname"], keep="first")
        df.reset_index(drop=True, inplace=True)
        st.success("✅ All images processed!")

        # Показать таблицу
        st.dataframe(df)

        # Скачивание Excel
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="⬇️ Download Excel",
            data=output,
            file_name="players_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
