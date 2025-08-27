from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import whisper
import soundfile as sf
import re
import os
import tempfile

app = FastAPI()

# Whisper 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

# 오디오 길이 체크
def is_audio_valid(file_path, min_duration=0.5):
    try:
        f = sf.SoundFile(file_path)
        duration = len(f) / f.samplerate
        return duration >= min_duration
    except:
        return True

# 반복 패턴, 의미 없는 텍스트 필터링
def clean_stt_text(text):
    text = text.strip()

    # 한 글자 이하 or 비어있음
    if len(text) <= 1:
        return ""

    # 같은 단어 반복 제거
    words = text.split()
    if len(set(words)) == 1 and len(words) > 5:
        return ""

    # 한글/영문 없는 경우 (주석 처리 또는 삭제)
    # if not re.search(r"[가-힣a-zA-Z]", text):
    #     return ""

    return text

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # 1. 오디오 길이 체크
        if not is_audio_valid(tmp_path, 0.5):
            os.remove(tmp_path)
            return JSONResponse({"text": ""})

        # 2. Whisper 추론
        result = model.transcribe(tmp_path, verbose=False)

        # no_speech_prob 확인 (첫 segment 기준)
        if result.get("segments") and result["segments"][0]["no_speech_prob"] > 0.8:
            os.remove(tmp_path)
            return JSONResponse({"text": ""})

        text = result.get("text", "")

        # 3. 후처리 필터링
        cleaned = clean_stt_text(text)

        # 인식된 텍스트를 터미널에 출력하는 코드 추가
        print(f"인식된 텍스트: {cleaned}")

        os.remove(tmp_path)
        return JSONResponse({"text": cleaned})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok"}


'''
cd /data/bootcamp/bootcamp/work_ID1/apps.250728_copy/servers/asr
conda activate whisper-server
CUDA_VISIBLE_DEVICES=2 uvicorn whisper_server:app --port 9000
'''
