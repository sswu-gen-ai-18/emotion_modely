# emotion_infer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 이 파일 기준으로 모델 폴더 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "kobert_emotion_final")

print("[emotion_infer] MODEL_DIR:", MODEL_DIR)
print("[emotion_infer] FILES:", os.listdir(MODEL_DIR))

# 1) 토크나이저: HuggingFace monologg/kobert 사용
tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True,
)

# 2) 모델: 네가 fine-tune 한 로컬 모델 사용
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()

# 3) id2label 설정 (config에 있으면 그걸 사용)
if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    id2label = {0: "anger", 1: "sad", 2: "fear"}

print("[emotion_infer] ID2LABEL:", id2label)

def predict_emotion(text: str):
    """한 문장의 감정을 (라벨, 점수)로 반환"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score, idx = torch.max(probs, dim=1)

    label = id2label[int(idx.item())]
    return label, float(score)
