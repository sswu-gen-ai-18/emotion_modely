# run_aihub_row_emotion_stream_fast.py

import os
import glob
import csv
import json
from emotion_infer import tokenizer, model, id2label
import torch

TRAIN_LABEL_DIR = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/1.Training/라벨링데이터_231222_add"

SPLITS = [("train", TRAIN_LABEL_DIR)]

OUTPUT_CSV = "/Users/ijiho/Desktop/callcenter_customer_emotions_train_stream_fast.csv"

BATCH_SIZE = 64    # 메모리 안전한 미니배치 사이즈


# ───────────────────────────────────────────────
# 텍스트 추출
# ───────────────────────────────────────────────
def extract_text(row):
    speaker = (row.get("화자") or "").strip()

    if speaker == "고객":
        candidates = ["고객질문(요청)", "고객답변"]
    else:
        candidates = ["상담사질문(요청)", "상담사답변"]

    for k in candidates:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


# ───────────────────────────────────────────────
# Batch 예측
# ───────────────────────────────────────────────
def batch_predict(text_list):
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score, idx = torch.max(probs, dim=1)

    results = []
    for i, s in zip(idx, score):
        results.append((id2label[int(i.item())], float(s)))
    return results


# ───────────────────────────────────────────────
# split 처리
# ───────────────────────────────────────────────
def process_split(split_name, folder):
    print(f"\n[INFO] Processing {split_name} ...")

    json_list = sorted(glob.glob(os.path.join(folder, "*.json")))
    print(f"  Found {len(json_list)} json files\n")

    results = []
    for json_path in json_list:
        fname = os.path.basename(json_path)
        print(f"  - Processing file: {fname}")

        # JSON을 전체 로드하지 않고 스트리밍 처리
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        batch_texts = []
        batch_info = []

        conv_counter = {}

        for row in data:
            speaker = (row.get("화자") or "").strip()
            if speaker != "고객":
                continue

            text = extract_text(row)
            if not text:
                continue

            conv_id = row.get("대화셋일련번호")
            if not conv_id:
                continue

            # 고객 발화 번호 증가
            conv_counter.setdefault(conv_id, 0)
            conv_counter[conv_id] += 1
            turn_index = conv_counter[conv_id]

            # 문장번호
            try:
                raw_turn = int(row.get("문장번호", 0))
            except:
                raw_turn = 0

            domain = row.get("도메인", "")
            category = row.get("카테고리", "")

            # 배치 리스트에 저장
            batch_texts.append(text)
            batch_info.append((fname, conv_id, domain, category, turn_index, raw_turn, speaker, text))

            # ───── 배치 단위로 모델에 넣기 ─────
            if len(batch_texts) >= BATCH_SIZE:
                preds = batch_predict(batch_texts)
                for info, (emo, score) in zip(batch_info, preds):
                    results.append({
                        "file": info[0],
                        "call_id": info[1],
                        "domain": info[2],
                        "category": info[3],
                        "customer_turn_index": info[4],
                        "raw_turn_index": info[5],
                        "speaker": info[6],
                        "text": info[7],
                        "emotion": emo,
                        "score": score,
                    })
                batch_texts = []
                batch_info = []

        # ───── 남은 배치 처리 ─────
        if batch_texts:
            preds = batch_predict(batch_texts)
            for info, (emo, score) in zip(batch_info, preds):
                results.append({
                    "file": info[0],
                    "call_id": info[1],
                    "domain": info[2],
                    "category": info[3],
                    "customer_turn_index": info[4],
                    "raw_turn_index": info[5],
                    "speaker": info[6],
                    "text": info[7],
                    "emotion": emo,
                    "score": score,
                })

        print(f"    → Completed {fname}")

    return results


# ───────────────────────────────────────────────
# main
# ───────────────────────────────────────────────
def main():
    rows = []

    for split, folder in SPLITS:
        rows.extend(process_split(split, folder))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "call_id", "domain", "category",
            "customer_turn_index", "raw_turn_index", "speaker", "text",
            "emotion", "score",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Saved {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
