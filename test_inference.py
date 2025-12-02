import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) 이 파일이 있는 위치 기준으로 모델 폴더 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "kobert_emotion_final")

print("MODEL_DIR:", MODEL_DIR)
print("FILES:", os.listdir(MODEL_DIR))

# 2) 토크나이저는 HuggingFace에서 원본 monologg/kobert 사용
tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True,
)

# 3) 모델은 네가 fine-tune 한 로컬 폴더에서 로드
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()

# 4) label 매핑 (config에 저장돼 있으면 거기서 읽고, 아니면 기본값 사용)
if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    id2label = {0: "anger", 1: "sad", 2: "fear"}

print("ID2LABEL:", id2label)

def predict(text: str):
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

if __name__ == "__main__":
    samples = [
        "배송이 너무 늦어요",
        "환불 받고 싶은데요",
        "진짜 화나 죽겠어요",
        "요즘 너무 불안하고 걱정돼요",
    ]
    for s in samples:
        print(s, "->", predict(s))
