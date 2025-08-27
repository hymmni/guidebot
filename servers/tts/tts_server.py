import os
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from melo.api import TTS

# 로깅 설정 (LLM 서버 스타일과 통일)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

OUTPUT_DIR = "/data/bootcamp/bootcamp/work_ID1/apps.250728_copy/melotts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = TTS(language='KR', device='cuda:0')
speaker_ids = model.hps.data.spk2id

@app.get("/health")
def health():
    try:
        test_path = os.path.join(OUTPUT_DIR, "test.wav")
        model.tts_to_file("테스트", speaker_ids["KR"], test_path, speed=1.0)
        if os.path.exists(test_path):
            os.remove(test_path)  # 생성 확인 후 삭제
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.exception("[x] TTS health check failed")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)

@app.post("/tts")
async def generate(request: Request):
    data = await request.json()
    text = data.get("text")
    speed = data.get("speed")

    if not text:
        logger.warning("[!] text 필드 없음")
        return JSONResponse({"error": "text is required"}, status_code=400)

    uid = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, f"{uid}.wav")

    logger.info(f"TTS 생성 요청 - 텍스트: {text} → 파일: {output_path}")
    await run_in_threadpool(model.tts_to_file, text, speaker_ids["KR"], output_path, speed)

    return JSONResponse({"audio_path": f"{uid}.wav"})

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        logger.warning(f"파일 없음: {filename}")
        return JSONResponse({"error": "file not found"}, status_code=404)

    logger.info(f"WAV 파일 전송: {filename}")
    return FileResponse(path, media_type="audio/wav")

'''
cd /data/bootcamp/bootcamp/work_ID1/apps.250728_copy/servers/tts
conda activate tts-server
CUDA_VISIBLE_DEVICES=3 uvicorn tts_server:app --port 9200
'''
