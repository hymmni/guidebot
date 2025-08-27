from fastapi import FastAPI, WebSocket, UploadFile, File, Body, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import uuid
import shutil
import os, json, asyncio
import requests

# ===============================
# 앱 설정
# ===============================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# 예) static/maps/floor1.png, static/maps/floor2.png 등이 존재

MAP_FILES = {
    "default": "venue_map.png",
    "1f": "floor1.png",
    "2f": "floor2.png",
}
def resolve_map_url(key: str | None) -> str:
    k = (key or "default").lower().replace(" ", "")
    fname = MAP_FILES.get(k, MAP_FILES["default"])
    return f"/static/maps/{fname}"
    
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

def run_tts(text: str, speed: float) -> str:
    response = requests.post(f"{TTS_SERVER_URL}/tts", json={"text": text, "speed": speed})
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
async def kiosk_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # 클라이언트에서 오는 메시지(JSON). 예:
            # 1) 하트비트: {"type":"ping"}
            # 2) 업로드 완료: {"file_id":"..."} 또는 {"type":"audio_uploaded","file_id":"..."}
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 텍스트가 JSON이 아니면 무시
                continue

            # 1) 하트비트 처리
            if data.get("type") == "ping":
                # 필요하면 pong 회신
                # await ws.send_json({"type": "pong"})
                continue

            # 2) 오디오 처리 트리거
            file_id = data.get("file_id")
            if not file_id and data.get("type") == "audio_uploaded":
                file_id = data.get("file_id")

            if not file_id:
                # 기타 메시지는 무시
                continue

            # ==== STT ====
            audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
            await ws.send_json({"stage": "status", "text": "음성 인식 중..."})
            try:
                stt_text = await run_in_threadpool(run_stt, audio_path)
            except Exception as e:
                await ws.send_json({"stage": "llm", "text": "음성 인식에 실패했어요. 다시 한 번 말씀해 주세요."})
                # 상황에 따라 바로 다음 턴 대기로 끝
                continue

            await ws.send_json({"stage": "stt", "text": stt_text})

            # ==== LLM 의도/플랜 ====
            await ws.send_json({"stage": "status", "text": "의도 분석 중..."})
            try:
                result = await run_in_threadpool(run_llm_intent, stt_text)
                if not isinstance(result, dict):
                    result = {}
            except Exception as e:
                result = {}

            # ==== 액션 해석 (지도 표시 등) ====
            for act in result.get("actions", []):
                svc  = act.get("service")
                name = act.get("name")
                params = act.get("params", {}) or {}

                # 지도/이미지 보여주기
                if svc == "INFO" and name in ("show_image", "show_map", "map"):
                    url = params.get("url")
                    if not url:
                        key = params.get("map_key")  # "1f", "2f" 등
                        url = resolve_map_url(key)
                    await ws.send_json({"stage": "ui", "type": "show_map", "url": url})

            # ==== 말할 응답 ====
            speak_text = (
                result.get("speak")
                or result.get("reply")
                or "알겠습니다."
            )
            await ws.send_json({"stage": "llm", "text": speak_text})

            # ==== TTS ====
            try:
                # run_tts는 파일명(또는 경로)을 반환한다고 가정
                tts_filename = await run_in_threadpool(run_tts, speak_text, 1.0)
                await ws.send_json({"stage": "tts", "audio_url": f"/ttsaudio/{tts_filename}"})
            except Exception:
                # TTS 실패해도 텍스트 응답은 이미 보냈으므로 조용히 무시
                pass

    except WebSocketDisconnect:
        # 정상 종료
        pass


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

class TTSRequest(BaseModel):
    text: str
    speed: float

@app.post("/request-response")
def request_tts(request: TTSRequest):
    tts_filename = run_tts(request.text, request.speed)
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
