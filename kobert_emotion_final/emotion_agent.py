# kobert_emotion_final/agents/emotion_agent.py
from emotion_infer import predict_emotion

class EmotionAgent:
    def predict(self, text: str) -> dict:
        """
        한 문장(text)에 대한 감정 예측을 dict로 반환
        """
        label, score = predict_emotion(text)
        return {
            "emotion_label": label,
            "emotion_score": score,
        }
