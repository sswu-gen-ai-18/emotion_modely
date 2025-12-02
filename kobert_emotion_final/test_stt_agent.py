# kobert_emotion_final/test_stt_agent.py
import os
from agents.stt_agent import STTAgent

def main():
    # 테스트용 음성 파일 하나 경로 넣기 (지금 Validation 쇼핑 폴더에 있으니까 그 중 하나)
    audio_path = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/2.Validation/원천데이터_220125_add/쇼핑/배송/쇼핑_8173.m4a"

    print("[TEST] audio_path:", audio_path)
    print("[EXISTS?]", os.path.exists(audio_path))

    stt = STTAgent(device="mps")  # 안 되면 "cpu"
    text = stt.run(audio_path)

    print("\n[STT RESULT]")
    print(text)

if __name__ == "__main__":
    main()
