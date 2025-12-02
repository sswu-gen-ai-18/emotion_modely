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

# 🔹 여기서 emotion_infer의 "발화 리스트용" 함수 가져오기
from emotion_infer import predict_emotions_by_utterance


if __name__ == "__main__":
    # 1) 기존처럼 한 문장씩 테스트
    samples = [
        "배송이 너무 늦어요",
        "환불 받고 싶은데요",
        "진짜 화나 죽겠어요",
        "요즘 너무 불안하고 걱정돼요",
    ]
    print("=== 단일 문장 테스트 ===")
    for s in samples:
        print(s, "->", predict(s))

    # 2) 발화 리스트(대화)로 테스트
    print("\n=== 발화별(고객 발화만) 감정 테스트 ===")
    conversation = [
        {"speaker": "customer", "text": "저 오늘 결제 내역이 이상해서요.", "turn": 1},
        {"speaker": "agent",    "text": "어떤 점이 이상하신가요?",       "turn": 2},
        {"speaker": "customer", "text": "두 번 결제된 것 같아요.",       "turn": 3},
        {"speaker": "agent",    "text": "확인해 보겠습니다.",           "turn": 4},
        {"speaker": "customer", "text": "이런 일이 자꾸 생기면 너무 화나요.", "turn": 5},
    ]

    # ⚠️ customer_tag는 네 데이터에 맞춰서 "customer" 또는 "고객"으로 맞춰줘야 해
    results = predict_emotions_by_utterance(
        conversation,
        speaker_key="speaker",
        text_key="text",
        customer_tag="customer",  # AI-Hub 원천데이터에서 고객이 어떻게 표기돼 있는지에 맞추기
    )

    for r in results:
        print(
            f"{r['customer_turn_index']}번째 고객 발화 "
            f"(전체 turn {r['raw_turn_index']}): \"{r['text']}\""
            f" -> 감정: {r['emotion']} (score={r['score']:.3f})"
        )
