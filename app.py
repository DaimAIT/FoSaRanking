import streamlit as st
import cv2
import pandas as pd
import numpy as np
import re
import gc
from paddleocr import PaddleOCR
from io import BytesIO

# Загружаем OCR один раз и кэшируем (не пересоздаём при каждом запуске)
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=False, lang="en", show_log=False)  # use_angle_cls=False снижает память

ocr = load_ocr()

# Фикс OCR ошибок (оставляем как есть)
fix_map = {
    # ...
}

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.strip()
    for k, v in fix_map.items():
        txt = txt.replace(k, v)
    return txt

def crop_table_body(img, top=0.0, bottom=0.0):
    """Обрезка лишних частей сверху/снизу для экономии памяти"""
    h, w = img.shape[:2]
    y1 = int(h * top)
    y2 = int(h * (1 - bottom))
    return img[y1:y2, :]

def preprocess_image(img):
    """Препроцессинг изображения для OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def parse_boxes(ocr_res):
    """Парсим боксы OCR"""
    return [(np.mean([p[0][0] for p in line[0]]), 
             np.mean([p[0][1] for p in line[0]]), 
             line[1][0]) for line in ocr_res]

def extract_rows_from_image(img):
    # минимизируем хранение промежуточных массивов
    img = crop_table_body(img, 0.24, 0.20)
    img = preprocess_image(img)

    h, w = img.shape[:2]
    ocr_res = ocr.ocr(img, cls=False)[0]  # cls=False тоже экономит память
    boxes = parse_boxes(ocr_res)

    left_bound = w * 0.14
    right_bound = w * 0.76

    # Разбивка
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
                clan_name = m.group(2).strip()

        rows.append({
            "Nickname": nick,
            "Clan tag": clan_tag,
            "Clan": clan_name,
            "Points": sc
        })

    # сортировка по очкам → присваиваем Rank
    rows.sort(key=lambda r: r["Points"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["Rank"] = i

    # чистим память после обработки
    del img, ocr_res, boxes, left, center, right, scores
    gc.collect()

    return rows

def process_images(uploaded_files):
    all_rows = []
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rows = extract_rows_from_image(img)
        all_rows.extend(rows)

        # чистим картинку из памяти
        del img, file_bytes, rows
        gc.collect()

    df = pd.DataFrame(all_rows, dtype="string")  # экономим память на хранении
    return df

# Streamlit UI
st.title("FoSa Ranking OCR Parser")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    df = process_images(uploaded_files)
    st.dataframe(df)

    # Скачивание Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    st.download_button("Download Excel", output.getvalue(), "results.xlsx")
