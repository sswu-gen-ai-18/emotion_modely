# kobert_emotion_final/test_pipeline_agents.py

import os
import sys

# === 1) agents 폴더를 sys.path에 추가 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../kobert_emotion_final
AGENT_DIR = os.path.join(BASE_DIR, "agents")                   # .../kobert_emotion_final/agents

if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from stt_agent import STTAgent
from emotion_agent import EmotionAgent


def process_audio(audio_path: str):
    """
    음성 파일 1개에 대해:
      1) STT로 전체 텍스트 뽑고
      2) 그 텍스트에 대한 감정 1개 예측
    """
    stt = STTAgent(device="mps")       # 안 되면 "cpu"
    emotion_agent = EmotionAgent()

    text = stt.run(audio_path)
    emotion = emotion_agent.predict(text)

    return {
        "text": text,
        "emotion": emotion,
    }


def main():
    audio_path = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/2.Validation/원천데이터_220125_add/쇼핑/배송/쇼핑_8173.m4a"

    print("[TEST] audio_path:", audio_path)
    print("[EXISTS?]", os.path.exists(audio_path))

    result = process_audio(audio_path)

    print("\n[TEXT PREVIEW]")
    print(result["text"][:200], "...")
    print("\n[EMOTION]")
    print(result["emotion"])


if __name__ == "__main__":
    main()
