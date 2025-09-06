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
    denoised = cv2.medianBlur(enhanced,3)
    return denoised  # без бинаризации
def extract_rows_from_image(img):
    img = crop_table_body(img, 0.24, 0.20)
    img = preprocess_image(img)

    h, w = img.shape[:2]
    
    ocr_res = ocr.ocr(img, cls=True)[0]
    boxes = parse_boxes(ocr_res)

    # Центр ников = 14–76% ширины
    left_bound = w * 0.2
    right_bound = w * 0.75

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
        # берём все боксы из center рядом по Y
        same_line = [c for c in center if abs(c[1] - sy) < h * 0.05]
        if not same_line:
            continue

        # сортируем слева направо
        same_line.sort(key=lambda c: c[0])

        # Ник = первый бокс без [TAG]
        nick_box = next((t for x, y, t in same_line if not t.strip().startswith("[")), None)
        clan_box = next((t for x, y, t in same_line if t.strip().startswith("[")), None)

        # Чистим текст
        nick = clean_text(nick_box) if nick_box else "N/A"
        clan_tag, clan_name = "FoSa", "Forgotten Saga"

        if clan_box:
            m = re.search(r"\[([^\]]+)\]\s*(.*)", clan_box)
            if m:
                clan_tag = m.group(1)
                clan_name = m.group(2).strip() or KNOWN_TAGS.get(clan_tag, "N/A")

        # Хак: если ник оканчивается на число (как Doomlord 13), не считать его клан-тегом
        if re.search(r"\s\d+$", nick):
            nick = nick

        rows.append({
            "Rank": next((r for ry, r in ranks if abs(ry - sy) < h * 0.04), i + 1),
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        img_path = "test.jpg"
        print(f"[DEBUG] Использую тестовый файл {img_path}")
    else:
        img_path = sys.argv[1]

    img = cv2.imread(img_path)
    rows = extract_rows_from_image(img)

    print("\nРаспознанные строки:")
    for r in rows:
        print(r)

    # Для удобства покажем DataFrame
    df = pd.DataFrame(rows)
    print("\nDataFrame:")
    print(df)

    # Визуализация OCR
    img_proc = crop_table_body(img, 0.24, 0.20)
    img_proc = preprocess_image(img_proc)
    h, w = img_proc.shape[:2]
    ocr_res = ocr.ocr(img_proc, cls=True)[0]
    boxes = parse_boxes(ocr_res)

    left_bound = w * 0.20
    right_bound = w * 0.75
    left   = [(x, y, t) for x, y, t in boxes if x < left_bound]
    center = [(x, y, t) for x, y, t in boxes if left_bound <= x <= right_bound]
    right  = [(x, y, t) for x, y, t in boxes if x > right_bound]

    vis = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR)

    def draw_points(data, color):
        for x, y, t in data:
            cv2.circle(vis, (int(x), int(y)), 6, color, -1)
            cv2.putText(vis, t, (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    draw_points(left,   (0, 0, 255))
    draw_points(center, (0, 255, 0))
    draw_points(right,  (255, 0, 0))

    cv2.imshow("OCR positions (all)", vis)
    cv2.imshow("Left",   img_proc[:, :int(left_bound)])
    cv2.imshow("Center", img_proc[:, int(left_bound):int(right_bound)])
    cv2.imshow("Right",  img_proc[:, int(right_bound):])

    print("\nНажми любую клавишу в окне изображения для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
