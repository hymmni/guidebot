# tts_server.py (교체/업데이트 부분)
import os, uuid, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from melo.api import TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

OUTPUT_DIR = "/data/bootcamp/bootcamp/work_ID1/apps.esppo/melotts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 언어별 모델 준비 (VRAM 여유가 있어야 함)
MODELS = {
    "KR": TTS(language="KR", device="cuda:0"),
    "EN": TTS(language="EN", device="cuda:0"),
    "JP": TTS(language="JP", device="cuda:0"),
}
SPEAKERS = {k: m.hps.data.spk2id for k, m in MODELS.items()}
LANG_MAP = {"ko": "KR", "en": "EN", "ja": "JP"}

@app.get("/health")
def health():
    try:
        # 간단 생성 확인(한국어)
        test = os.path.join(OUTPUT_DIR, "test.wav")
        MODELS["KR"].tts_to_file("테스트", SPEAKERS["KR"]["KR"], test, speed=1.0)
        if os.path.exists(test):
            os.remove(test)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.exception("[x] TTS health check failed")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)

@app.post("/tts")
async def generate(request: Request):
    data  = await request.json()
    text  = (data.get("text") or "").strip()
    speed = float(data.get("speed") or 1.0)
    lang  = (data.get("lang") or "ko").lower()

    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    code = LANG_MAP.get(lang, "KR")
    model = MODELS.get(code, MODELS["KR"])
    spkid = SPEAKERS[code][code]  # 스피커 키: "KR"/"EN"/"JP"

    uid = str(uuid.uuid4())
    out = os.path.join(OUTPUT_DIR, f"{uid}.wav")
    logger.info(f"TTS [{lang}->{code}] '{text[:30]}...' -> {out}")

    await run_in_threadpool(model.tts_to_file, text, spkid, out, speed)
    # main.py 호환: filename/audio_path 어떤 키든 인식함
    return JSONResponse({"filename": f"{uid}.wav", "audio_path": out})

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")
