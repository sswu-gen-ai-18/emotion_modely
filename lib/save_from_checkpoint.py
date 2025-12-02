# save_from_checkpoint.py

import os
import json
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) 이미 학습된 체크포인트 경로
SRC_DIR = "../.venv/lib/kobert_emotion_out/checkpoint-4638"  # 왼쪽 트리에 보이던 폴더
MODEL_SAVE_DIR = "../.venv/lib/kobert_emotion_final/kobert_emotion_final"  # 최종 모델 폴더

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2) 체크포인트에서 모델 불러오기
print("🔁 Loading trained model from:", os.path.abspath(SRC_DIR))
model = AutoModelForSequenceClassification.from_pretrained(
    SRC_DIR,
    trust_remote_code=True,
)

# 3) 우리가 쓸 최종 폴더로 저장
model.save_pretrained(MODEL_SAVE_DIR)
print("✅ Model saved to:", os.path.abspath(MODEL_SAVE_DIR))

# 4) KoBERT 토크나이저는 허깅페이스에서 가져와서 수동 저장
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# ---- KoBERT tokenizer manual save ----
# vocab.txt
vocab_path = tokenizer.vocab_file
shutil.copy(vocab_path, os.path.join(MODEL_SAVE_DIR, "vocab.txt"))

# sentencepiece 모델 파일 있으면 같이 복사
if hasattr(tokenizer, "sp_model_file"):
    shutil.copy(
        tokenizer.sp_model_file,
        os.path.join(MODEL_SAVE_DIR, "tokenizer.model")
    )

# tokenizer_config.json
with open(os.path.join(MODEL_SAVE_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer.init_kwargs, f, ensure_ascii=False, indent=2)

# special_tokens_map.json
with open(os.path.join(MODEL_SAVE_DIR, "special_tokens_map.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer.special_tokens_map, f, ensure_ascii=False, indent=2)

print("📌 KoBERT tokenizer saved manually.")

# 5) label_map.json (anger/sad/fear 매핑)
label2id = {"anger": 0, "sad": 1, "fear": 2}
id2label = {v: k for k, v in label2id.items()}

with open(os.path.join(MODEL_SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print("📄 label_map.json saved.")
print("🎉 ALL DONE")
