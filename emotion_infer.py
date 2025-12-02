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


def predict_emotions_by_utterance(
    utterances,
    speaker_key: str = "speaker",
    text_key: str = "text",
    customer_tag: str = "고객",
):
    """
    발화 리스트 단위로 감정 분석해 주는 함수.

    utterances 예시 형식:
    [
        {"speaker": "고객", "text": "저 오늘 결제 내역이 이상해서요.", "turn": 1},
        {"speaker": "상담사", "text": "어떤 점이 이상하신가요?",         "turn": 2},
        {"speaker": "고객", "text": "두 번 결제된 것 같아요.",         "turn": 3},
        ...
    ]

    speaker_key : 화자 정보 key 이름 (기본 "speaker")
    text_key    : 실제 텍스트가 들어 있는 key 이름 (기본 "text")
    customer_tag: 고객을 나타내는 값 (예: "고객", "customer", "user" 등)
    """

    results = []
    customer_turn_index = 1  # 1번째 고객 발화, 2번째 고객 발화...

    for utt in utterances:
        speaker = (utt.get(speaker_key) or "").strip()
        if speaker != customer_tag:
            # 상담사/시스템 발화는 감정 분석 스킵
            continue

        text = (utt.get(text_key) or "").strip()
        if not text:
            continue

        label, score = predict_emotion(text)

        results.append({
            "customer_turn_index": customer_turn_index,  # 고객 기준 n번째 발화
            "raw_turn_index": utt.get("turn"),          # 전체 대화 기준 turn (있으면)
            "speaker": speaker,
            "text": text,
            "emotion": label,
            "score": score,                             # 예측 확률(신뢰도)
        })

        customer_turn_index += 1

    return results


def get_last_customer_emotion(
    conversation,
    speaker_key: str = "speaker",
    text_key: str = "text",
    customer_tag: str = "고객",
):
    """
    대화(conversation) 전체에서 '마지막 고객 발화'의 감정만 가져오는 헬퍼 함수.

    conversation 예시:
    [
        {"speaker": "고객", "text": "배송이 너무 늦어요", "turn": 1},
        {"speaker": "상담사", "text": "확인 도와드리겠습니다", "turn": 2},
        {"speaker": "고객", "text": "진짜 화나 죽겠어요", "turn": 3},
    ]

    반환 예시:
    {
        "customer_turn_index": 2,
        "raw_turn_index": 3,
        "speaker": "고객",
        "text": "진짜 화나 죽겠어요",
        "emotion": "anger",
        "score": 0.93
    }
    """
    results = predict_emotions_by_utterance(
        conversation,
        speaker_key=speaker_key,
        text_key=text_key,
        customer_tag=customer_tag,
    )

    if not results:
        return None

    # 가장 마지막 고객 발화의 감정 정보
    return results[-1]
