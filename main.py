# main.py — Kiosk backend: WS 파이프라인 + 멀티랭 TTS 연동
import os, uuid, json, shutil, glob, logging, requests
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kiosk")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── 디렉터리/정적 리소스
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 유물/이미지: ./artifacts 를 /artifacts 로 서빙 (금관 파일 등)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
if os.path.isdir(ARTIFACT_DIR):
    app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR), name="artifacts")
else:
    log.warning("ARTIFACT_DIR not found: %s", ARTIFACT_DIR)

# 업로드/음성 출력 폴더
UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", "/tmp/audio_kiosk")
TTS_OUT_DIR = os.path.join(BASE_DIR, "melotts_output")  # 기존 구조 유지
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_OUT_DIR, exist_ok=True)
app.mount("/ttsaudio", StaticFiles(directory=TTS_OUT_DIR), name="ttsaudio")

# 외부 마이크로서비스
STT_SERVER_URL    = os.environ.get("STT_URL",   "http://localhost:9000")
LLM_SERVER_URL    = os.environ.get("LLM_URL",   "http://localhost:9100")
TTS_SERVER_URL    = os.environ.get("TTS_URL",   "http://localhost:9200")
NOTIFY_SERVER_URL = os.environ.get("NOTIFY_URL","http://localhost:9300")
NAV_SERVER_URL    = os.environ.get("NAV_URL",   "")

# ── 지도 매핑
MAP_FILES = {
    "default": "venue_map.png",
    "b1": "b1.png",
    "1f": "floor1.png",
    "2f": "floor2.png",
    "3f": "floor3.png",
}
def resolve_map_url(key: Optional[str]) -> str:
    k = (key or "default").lower().replace(" ", "")
    fname = MAP_FILES.get(k, MAP_FILES["default"])
    fs_path = os.path.join(STATIC_DIR, "maps", fname)
    if not os.path.exists(fs_path):
        log.warning("map not found: %s (fallback default)", fs_path)
        return f"/static/maps/{MAP_FILES['default']}"
    return f"/static/maps/{fname}"

def _first_ext_path(file_id: str) -> Optional[str]:
    pats = glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}.*"))
    return pats[0] if pats else None

def guess_floor_key(text: Optional[str]) -> str:
    t = (text or "").lower().replace(" ", "")
    if ("b1" in t) or ("지하1" in t): return "b1"
    if ("2층" in t) or ("이층" in t) or ("2f" in t): return "2f"
    if ("1층" in t) or ("일층" in t) or ("1f" in t) or ("로비" in t) or ("lobby" in t): return "1f"
    return "default"

# ── 헬스체크
@app.get("/health")
def health_check():
    results = {}
    for name, url in [("stt", STT_SERVER_URL), ("llm", LLM_SERVER_URL), ("tts", TTS_SERVER_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            results[name] = (r.status_code == 200)
        except:
            results[name] = False
    return JSONResponse(content=results)

# ── 외부 호출
def run_stt(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        r = requests.post(f"{STT_SERVER_URL}/stt", files={"file": f}, timeout=60)
    r.raise_for_status()
    return (r.json() or {}).get("text", "")

def run_llm_intent(user_text: str) -> Dict[str, Any]:
    r = requests.post(f"{LLM_SERVER_URL}/intent", json={"text": user_text}, timeout=20)
    r.raise_for_status()
    return r.json()

def tts_make_audio_url(text: str, speed: float = 1.0, lang: str = "ko") -> str:
    try:
        r = requests.post(
            f"{TTS_SERVER_URL}/tts",
            json={"text": text, "speed": speed, "lang": lang},
            timeout=30
        )
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        if data.get("audio_url"):
            return data["audio_url"]
        if data.get("filename"):
            return f"/ttsaudio/{data['filename']}"
        if data.get("audio_path"):
            return f"/ttsaudio/{os.path.basename(data['audio_path'])}"
    except Exception as e:
        log.warning("TTS error: %s", e)
    return ""

def run_notify_staff(payload: Optional[Dict[str, Any]] = None) -> bool:
    try:
        r = requests.post(f"{NOTIFY_SERVER_URL}/notify", json=payload or {"topic":"call_staff"}, timeout=5)
        return r.ok
    except Exception as e:
        log.warning("notify_staff failed: %s", e)
        return False

def run_nav_route(src: Optional[str], dst: Optional[str]) -> Dict[str, Any]:
    if dst and NAV_SERVER_URL:
        try:
            r = requests.post(f"{NAV_SERVER_URL}/route", json={"from":src, "to":dst}, timeout=5)
            if r.ok: return r.json()
        except Exception as e:
            log.warning("nav server failed, fallback: %s", e)
    steps = []
    if src: steps.append({"text": f"{src}에서 출발"})
    if dst:
        steps += [{"text": f"{dst}까지 직진 후 좌회전"}, {"text": "안내 표지판을 따라 50m 이동"}]
    return {"from": src, "to": dst, "steps": steps}

# ── 업로드 & TTS 프록시
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    mt  = (file.content_type or "").lower()
    ext = ".webm"
    if "wav" in mt: ext = ".wav"
    elif "ogg" in mt: ext = ".ogg"
    elif "mp3" in mt: ext = ".mp3"
    out_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id}

@app.post("/request-response")
async def request_response(req: Request):
    data = await req.json()
    text  = data.get("text", "")
    speed = float(data.get("speed", 1.0) or 1.0)
    lang  = (data.get("lang") or "ko")
    url = await run_in_threadpool(tts_make_audio_url, text, speed, lang)
    return JSONResponse({"audio_url": url})

# ── WebSocket 파이프라인
@app.websocket("/ws/kiosk")
async def kiosk_ws(ws: WebSocket):
    await ws.accept()
    session_lang = "ko"  # 기본 언어(한국어)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # 핑
            if data.get("type") == "ping":
                continue

            # 세션 언어 업데이트 헬퍼
            def update_lang_from_result(result: dict):
                nonlocal session_lang
                slot_lang = (result.get("slots") or {}).get("lang")
                if slot_lang:
                    session_lang = slot_lang  # ko/en/ja/zh

            # 공통 액션 처리
            async def handle_action(act: Dict[str, Any], result: Dict[str, Any]):
                nonlocal session_lang
                svc    = act.get("service")
                name   = act.get("name")
                params = (act.get("params") or {})

                if svc == "UI" and name == "choose_floor":
                    await ws.send_json({
                        "stage":"ui","type":"choose_floor",
                        "floors": params.get("floors") or ["b1"]+[f"{i}f" for i in range(1,11)],
                        "prompt": params.get("prompt","")
                    })

                if svc == "UI" and name == "choice":
                    await ws.send_json({
                        "stage":"ui","type":"choice",
                        "title": params.get("title",""),
                        "prompt": params.get("prompt",""),
                        "options": params.get("options",[])
                    })

                if svc == "INFO" and name in ("show_image","show_map","map"):
                    url = params.get("url") or resolve_map_url(params.get("map_key"))
                    await ws.send_json({"stage":"ui","type":"show_map","url": url})

                if svc == "INFO" and name in ("place_info","show_info"):
                    text = params.get("text") or result.get("speak") or ""
                    await ws.send_json({"stage":"ui","type":"show_info","html": text})

                if svc == "INFO" and name == "show_artifact":
                    await ws.send_json({"stage":"ui","type":"show_artifact",
                                        "title": params.get("title",""),
                                        "url": params.get("url",""),
                                        "desc": params.get("desc","")})

                if svc == "NAV" and name == "plan_route":
                    route = await run_in_threadpool(run_nav_route, params.get("from"), params.get("to"))
                    await ws.send_json({"stage":"ui","type":"nav_route","route": route})

                if svc == "CALL" and name in ("notify_staff","notify_security"):
                    ok = await run_in_threadpool(run_notify_staff, {"type":name, "meta":params})
                    await ws.send_json({"stage":"ui","type":"call_staff_ack","ok": bool(ok)})

                if svc in ("UI","INFO") and name in ("set_lang","change_lang","set_language"):
                    lang = params.get("lang") or (result.get("slots") or {}).get("lang") or "ko"
                    session_lang = lang
                    await ws.send_json({"stage":"ui","type":"set_lang","lang": lang})

                if svc == "TTS" and name == "speak_detail":
                    long_txt = (params.get("text") or "").strip()
                    if long_txt:
                        url = await run_in_threadpool(tts_make_audio_url, long_txt, 1.0, session_lang)
                        if url: await ws.send_json({"stage":"tts","audio_url": url})

            # 좌측 버튼 → 가짜 유저 발화
            if data.get("type") == "synthetic_text":
                stt_text = (data.get("text") or "").strip()
                if not stt_text: continue
                await ws.send_json({"stage":"stt","text": stt_text})
                await ws.send_json({"stage":"status","text":"의도 분석 중..."})
                try:
                    result = await run_in_threadpool(run_llm_intent, stt_text)
                except Exception:
                    result = {"intent":"UNKNOWN","actions":[],"speak":"이해하지 못했어요.","ask_user":"", "slots":{}, "confidence":0.2}

                update_lang_from_result(result)

                for act in result.get("actions", []):
                    await handle_action(act, result)

                speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
                await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})
                try:
                    url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0, session_lang)
                    if url: await ws.send_json({"stage":"tts","audio_url": url})
                except Exception:
                    pass
                continue

            # 층 선택 패널 클릭
            if data.get("type") == "select_floor":
                key = (data.get("map_key") or "").lower().strip()
                if key:
                    def floor_label(k: str) -> str:
                        if k.startswith("b") and k[1:].isdigit(): return f"지하 {int(k[1:])}층"
                        if k.endswith("f") and k[:-1].isdigit():  return f"{int(k[:-1])}층"
                        return "해당 층"
                    label = floor_label(key)
                    url   = resolve_map_url(key)
                    await ws.send_json({"stage":"ui","type":"show_map","url": url})
                    speak_text = f"{label} 지도를 보여드릴게요."
                    await ws.send_json({"stage":"llm","text": speak_text, "user_text": ""})
                    tts_url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0, session_lang)
                    if tts_url:
                        await ws.send_json({"stage":"tts","audio_url": tts_url})
                continue

            # 업로드 파이프라인
            file_id = data.get("file_id") or (data.get("type") == "audio_uploaded" and data.get("file_id"))
            if not file_id:
                continue

            upath = _first_ext_path(file_id)
            if not upath:
                err_text = "업로드된 음성을 찾지 못했습니다. 다시 말씀해 주세요."
                await ws.send_json({"stage":"llm","text": err_text, "user_text": ""})
                url = await run_in_threadpool(tts_make_audio_url, err_text, 1.0, session_lang)
                if url: await ws.send_json({"stage":"tts","audio_url": url})
                continue

            await ws.send_json({"stage":"status","text":"음성 인식 중..."})
            try:
                stt_text = await run_in_threadpool(run_stt, upath)
            except Exception:
                stt_text = ""

            if not stt_text:
                err_text = "음성 인식에 실패했어요. 다시 말씀해 주세요."
                await ws.send_json({"stage":"llm","text": err_text, "user_text": stt_text})
                url = await run_in_threadpool(tts_make_audio_url, err_text, 1.0, session_lang)
                if url: await ws.send_json({"stage":"tts","audio_url": url})
                continue

            await ws.send_json({"stage":"stt","text": stt_text})
            await ws.send_json({"stage":"status","text":"의도 분석 중..."})
            try:
                result = await run_in_threadpool(run_llm_intent, stt_text)
            except Exception:
                result = {"intent":"UNKNOWN","actions":[],"speak":"이해하지 못했어요.","ask_user":"", "slots":{}, "confidence":0.2}

            update_lang_from_result(result)

            for act in result.get("actions", []):
                await handle_action(act, result)

            speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
            await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})
            url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0, session_lang)
            if url:
                await ws.send_json({"stage":"tts","audio_url": url})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception("WebSocket fatal: %s", e)
        await ws.close(code=1011)

# 프론트 정적 파일
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
