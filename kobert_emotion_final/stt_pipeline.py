import os
import glob
import whisper
from emotion_infer import predict_emotion  # 감정분석 모듈

def find_child(parent, keyword):
    """parent 안에서 이름에 keyword가 들어간 하위 폴더를 찾아서 경로 반환"""
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if keyword in name and os.path.isdir(path):
            return path
    raise FileNotFoundError(f"'{keyword}' 를 포함한 폴더를 {parent} 안에서 찾지 못함")

DOWNLOADS = os.path.expanduser("~/Downloads")

# 단계별 탐색
step1 = find_child(DOWNLOADS, "022.")          # 022.민원(콜센터) 질의-응답 데이터
step2 = find_child(step1, "01.")              # 01.데이터
step3 = find_child(step2, "Validation")       # 2.Validation...
step4 = find_child(step3, "원천데이터")       # 원천데이터_220125_add  (여기까지 내려옴)
AUDIO_DIR = find_child(step4, "쇼핑")         # 쇼핑 폴더

print("DOWNLOADS:", DOWNLOADS)
print("STEP1    :", step1)
print("STEP2    :", step2)
print("STEP3    :", step3)
print("STEP4    :", step4)
print("AUDIO_DIR:", AUDIO_DIR)
print("AUDIO_DIR exists?:", os.path.isdir(AUDIO_DIR))


# 🔹 3) Whisper 모델 로드
device = "mps"   # 안 되면 "cpu"
whisper_model = whisper.load_model("small", device=device)

def stt_whisper(audio_path: str) -> str:
    print(f"\n[STT] {os.path.basename(audio_path)}")
    result = whisper_model.transcribe(
        audio_path,
        language="ko",
        fp16=False,   # 🔥 이 줄 추가!
    )
    return result["text"]

def main():
    # 🔹 4) os.walk로 m4a를 전부 찾기 (대소문자 무시)
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
        print("❗ m4a 파일을 찾지 못했습니다. 확장자나 경로를 다시 확인해줘.")
        return

    # 🔹 5) 몇 개만 STT + 감정분석
    for audio_path in audio_files[:5]:
        text = stt_whisper(audio_path)

        preview = text[:120].replace("\n", " ")
        print("[TEXT PREVIEW]:", preview, "..." if len(text) > 120 else "")

        label, score = predict_emotion(text)
        print("[EMOTION]:", label, f"(score={score:.4f})")

if __name__ == "__main__":
    main()
