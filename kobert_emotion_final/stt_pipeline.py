import os
import glob
import whisper
from emotion_infer import (
    predict_emotions_by_utterance,
    get_last_customer_emotion,
)

def find_child(parent, keyword):
    """parent 안에서 이름에 keyword가 들어간 하위 폴더를 찾아서 경로 반환"""
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if keyword in name and os.path.isdir(path):
            return path
    raise FileNotFoundError(f"'{keyword}' 를 포함한 폴더를 {parent} 안에서 찾지 못함")


DOWNLOADS = os.path.expanduser("~/Downloads")

# 단계별 탐색
step1 = find_child(DOWNLOADS, "022.")
step2 = find_child(step1, "01.")
step3 = find_child(step2, "Validation")
step4 = find_child(step3, "원천데이터")
AUDIO_DIR = find_child(step4, "쇼핑")

print("AUDIO_DIR:", AUDIO_DIR)
print("AUDIO_DIR exists?:", os.path.isdir(AUDIO_DIR))


# Whisper 모델 로드
device = "mps"  # 안 되면 "cpu"
whisper_model = whisper.load_model("small", device=device)


def stt_whisper(audio_path: str) -> str:
    print(f"\n[STT] {os.path.basename(audio_path)}")
    result = whisper_model.transcribe(
        audio_path,
        language="ko",
        fp16=False,
    )
    return result["text"]


def split_sentences_korean(text: str):
    """
    Whisper 결과를 단순 문장 단위로 분리하는 함수.
    '고객' 발화로 가정.
    """
    import re

    # 마침표, 느낌표, 물음표 뒤에서 split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 공백/빈 문장 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def main():
    # 다수의 m4a 찾기
    audio_files = []
    for root, dirs, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.lower().endswith(".m4a"):
                audio_files.append(os.path.join(root, f))

    print("\n찾은 음성 파일 개수:", len(audio_files))
    print("샘플 5개:")
    for p in audio_files[:5]:
        print(" -", p)

    if not audio_files:
        print("❗ m4a 파일이 없습니다.")
        return

    # STT + 감정
    for audio_path in audio_files[:3]:  # 일단 3개만
        # 1) STT
        text = stt_whisper(audio_path)
        print("[TEXT PREVIEW]:", text[:150], "...")

        # 2) 문장 단위로 split
        sentences = split_sentences_korean(text)

        # 3) emotion_infer가 이해하는 형태로 변환
        utterances = []
        turn = 1

        for s in sentences:
            utterances.append({
                "speaker": "고객",  # 원천데이터는 speaker 정보 없으니까 고객으로 가정
                "text": s,
                "turn": turn
            })
            turn += 1

        # 4) 전체 고객 감정
        all_emotions = predict_emotions_by_utterance(utterances, customer_tag="고객")

        print("\n[전체 고객 발화 감정]")
        for item in all_emotions:
            print(f" - {item['customer_turn_index']}번째: {item['emotion']} ({item['score']:.3f})  | {item['text']}")

        # 5) 마지막 고객 발화
        last = get_last_customer_emotion(utterances, customer_tag="고객")
        print("\n[마지막 고객 발화 감정]")
        print(last)


if __name__ == "__main__":
    main()
