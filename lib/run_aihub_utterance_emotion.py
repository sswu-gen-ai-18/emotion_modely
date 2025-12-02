# run_all_label_utter_emotion.py

import os
import glob
import json
import csv
from collections import defaultdict

from emotion_infer import predict_emotions_by_utterance


# 🔹 네 환경에 맞게 경로만 확인/수정
TRAIN_LABEL_DIR = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/1.Training/라벨링데이터_231222_add"
VAL_LABEL_DIR = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/2.Validation/라벨링데이터_231222_add"


SPLITS = [
    ("train", TRAIN_LABEL_DIR),
    ("val",   VAL_LABEL_DIR),
]

OUTPUT_CSV = "/Users/ijiho/Desktop/callcenter_customer_emotions_all.csv"


def get_text_for_row(row: dict) -> str:
    """
    한 row에서 실제 발화 텍스트를 뽑는 함수.
    화자(고객/상담사)에 따라 알맞은 필드에서 꺼낸다.
    """
    speaker = row.get("화자", "").strip()

    if speaker == "고객":
        candidates = ["고객질문(요청)", "고객반박", "QA"]
    else:  # 상담사
        candidates = ["상담사답변", "상담사다법", "QA"]

    for k in candidates:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def process_split(split_name: str, label_dir: str):
    """
    한 split(train 또는 val)의 폴더 안에 있는
    모든 .json 파일을 처리해서 결과 row 리스트를 리턴.
    """
    print(f"\n[INFO] Processing split={split_name}, dir={label_dir}")

    # 폴더 안의 모든 json 파일 찾기
    json_paths = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    print(f"  Found {len(json_paths)} json files")

    all_rows = []

    for json_path in json_paths:
        file_name = os.path.basename(json_path)
        print(f"  - {file_name}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # 최상단이 리스트라고 가정

        # 통화별로 발화 모으기
        dialogs = defaultdict(list)

        for row in data:
            conv_id = row.get("대화식별번호")
            if not conv_id:
                continue

            speaker = (row.get("화자") or "").strip()
            text = get_text_for_row(row)
            # 문장번호가 문자열일 수 있으니 int 변환
            try:
                turn = int(row.get("문장번호", 0))
            except ValueError:
                turn = 0

            # 도메인/카테고리도 같이 기록해두면 나중에 분석에 유리
            domain = row.get("도메인", "")
            category1 = row.get("카테고리1", "")

            dialogs[conv_id].append({
                "speaker": speaker,
                "text": text,
                "turn": turn,
                "domain": domain,
                "category1": category1,
            })

        # 각 통화별로 정렬 후 감정 분석
        for conv_id, utterances in dialogs.items():
            # turn 순으로 정렬
            utterances.sort(key=lambda x: x["turn"])

            # 고객 발화만 감정 분석 (상담사는 자동으로 스킵)
            results = predict_emotions_by_utterance(
                utterances,
                speaker_key="speaker",
                text_key="text",
                customer_tag="고객",   # 화자 값이 '고객'인 경우만 사용
            )

            # 결과 정리
            for r in results:
                # 해당 turn의 domain/category1 찾아오기
                # (utterances 리스트에서 raw_turn_index에 해당하는 것)
                meta = next(
                    (u for u in utterances if u["turn"] == r["raw_turn_index"]),
                    {"domain": "", "category1": ""}
                )

                all_rows.append({
                    "split": split_name,                       # train / val
                    "file": file_name,                         # 어떤 json에서 왔는지
                    "call_id": conv_id,                        # 대화식별번호
                    "domain": meta.get("domain", ""),
                    "category1": meta.get("category1", ""),
                    "customer_turn_index": r["customer_turn_index"],  # 1번째/2번째 고객 발화
                    "raw_turn_index": r["raw_turn_index"],            # 전체 발화 순서
                    "speaker": r["speaker"],                   # 항상 '고객'
                    "text": r["text"],
                    "emotion": r["emotion"],                   # anger / sad / fear
                    "score": r["score"],                       # 확률
                })

    return all_rows


def main():
    rows = []
    for split_name, label_dir in SPLITS:
        rows.extend(process_split(split_name, label_dir))

    # CSV 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "file",
                "call_id",
                "domain",
                "category1",
                "customer_turn_index",
                "raw_turn_index",
                "speaker",
                "text",
                "emotion",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
