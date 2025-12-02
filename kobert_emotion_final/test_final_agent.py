# kobert_emotion_final/test_final_agent.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../kobert_emotion_final
AGENT_DIR = os.path.join(BASE_DIR, "agents")               # .../kobert_emotion_final/agents

if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from final_agent import CallcenterAudioProcessor

if __name__ == "__main__":
    processor = CallcenterAudioProcessor()

    audio_path = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/2.Validation/원천데이터_220125_add/쇼핑/배송/쇼핑_8173.m4a"

    print("[TEST PATH]", audio_path, "exists=", os.path.exists(audio_path))

    result = processor.process_audio(audio_path)

    print("\n=== FINAL RESULT ===")
    print("📌 텍스트 일부:", result["text"][:120], "...")
    print("📌 감정:", result["emotion"])
