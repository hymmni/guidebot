from fastapi import FastAPI, WebSocket, UploadFile, File, Body
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import uuid
import shutil
import os
import requests

# ===============================
# 앱 설정
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/audio_kiosk"
TTS_SERVER_URL = "http://localhost:9200"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===============================
# 서버 헬스체크
# ===============================
@app.get("/health")
def health_check():
    results = {}
    try:
        r = requests.get("http://localhost:9000/health")
        results["stt"] = r.status_code == 200
    except:
        results["stt"] = False

    try:
        r = requests.get("http://localhost:9100/health")
        results["llm"] = r.status_code == 200
    except:
        results["llm"] = False

    try:
        r = requests.get("http://localhost:9200/health")
        results["tts"] = r.status_code == 200
    except:
        results["tts"] = False

    return JSONResponse(content=results)

# ===============================
# 외부 서버 호출 함수
# ===============================
def run_stt(audio_path: str):
    with open(audio_path, "rb") as f:
        response = requests.post("http://localhost:9000/stt", files={"file": f})
    return response.json()["text"]

def run_llm_intent(user_text: str) -> dict:
    r = requests.post("http://localhost:9100/intent", json={"text": user_text, "lang": "ko"})
    r.raise_for_status()
    return r.json()

def run_tts(text: str) -> str:
    response = requests.post(f"{TTS_SERVER_URL}/tts", json={"text": text, "speed": 1.0})
    if response.status_code != 200:
        raise RuntimeError("TTS 서버 오류")
    return response.json()["audio_path"]

# ===============================
# 파일 업로드 / TTS 파일 제공
# ===============================
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_id": file_id}

@app.get("/ttsaudio/{filename}")
def run_tts_audio(filename: str):
    response = requests.get(f"{TTS_SERVER_URL}/audio/{filename}")
    return Response(content=response.content, media_type="audio/wav")

# ===============================
# WebSocket 상호작용
# ===============================
@app.websocket("/ws/kiosk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    for stage in ["status", "stt", "llm", "tts"]:
        await websocket.send_json({"stage": stage, "text": ""} if stage != "tts" else {"stage": stage, "audio_url": ""})

    try:
        while True:
            data = await websocket.receive_json()
            file_id = data.get("file_id")
            audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

            # 1. STT
            await websocket.send_json({"stage": "status", "text": "음성 인식 중..."})
            stt_text = await run_in_threadpool(run_stt, audio_path)
            await websocket.send_json({"stage": "stt", "text": stt_text})

            # 2. Intent LLM
            await websocket.send_json({"stage": "status", "text": "의도 분석 중..."})
            plan = await run_in_threadpool(run_llm_intent, stt_text)
            speak_text = plan.get("speak", "") or "알겠습니다."
            await websocket.send_json({"stage": "llm", "text": speak_text})

            # 3. TTS
            await websocket.send_json({"stage": "status", "text": "음성 생성 중..."})
            tts_filename = await run_in_threadpool(run_tts, speak_text)
            audio_url = f"/ttsaudio/{tts_filename}"
            await websocket.send_json({"stage": "tts", "audio_url": audio_url})

            # 완료
            await websocket.send_json({"stage": "status", "text": "처리 완료"})
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

# ===============================
# 채팅 테스트용 엔드포인트
# ===============================
class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat_with_llm(chat_request: ChatRequest):
    try:
        plan = await run_in_threadpool(run_llm_intent, chat_request.text)
        return plan
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ===============================
# 인사말 TTS
# ===============================
@app.post("/greet")
def greet(text: str = Body("무엇을 도와드릴까요?")):
    tts_filename = run_tts(text)
    return {"audio_url": f"/ttsaudio/{tts_filename}"}

# ===============================
# 정적 파일 서비스
# ===============================
app.mount("/", StaticFiles(directory="static", html=True), name="static")



'''
cd /data/bootcamp/bootcamp/work_ID1/apps.250728_copy
conda activate app-backend
uvicorn main:app --port 8000
'''
