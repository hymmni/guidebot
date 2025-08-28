# main.py
import os, uuid, json, shutil, glob, logging, requests
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

# ===============================
# 로깅
# ===============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kiosk")

# ===============================
# 앱/경로/외부 서버 설정
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# 정적(웹)
STATIC_DIR  = os.path.join(BASE_DIR, "static")             # index.html, styles.css, maps/...
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 업로드/TTS
UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", "/tmp/audio_kiosk")
TTS_OUT_DIR = os.path.join(BASE_DIR, "melotts_output")     # TTS 파일 폴더
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_OUT_DIR, exist_ok=True)
app.mount("/ttsaudio", StaticFiles(directory=TTS_OUT_DIR), name="ttsaudio")  # 정적 서빙

# 외부 마이크로서비스
STT_SERVER_URL    = os.environ.get("STT_URL",   "http://localhost:9000")
LLM_SERVER_URL    = os.environ.get("LLM_URL",   "http://localhost:9100")
TTS_SERVER_URL    = os.environ.get("TTS_URL",   "http://localhost:9200")
NOTIFY_SERVER_URL = os.environ.get("NOTIFY_URL","http://localhost:9300")
NAV_SERVER_URL    = os.environ.get("NAV_URL",   "")  # 없으면 폴백

# 지도 파일 매핑( static/maps/ 의 실제 파일명과 일치 )
MAP_FILES = {
    "default": "venue_map.png",
    "1f": "floor1.png",
    "2f": "floor2.png",
    "b1": "b1.png",   # 있으면 사용
}
def resolve_map_url(key: Optional[str]) -> str:
    k = (key or "default").lower().replace(" ", "")
    fname = MAP_FILES.get(k, MAP_FILES["default"])
    fs_path = os.path.join(STATIC_DIR, "maps", fname)
    if not os.path.exists(fs_path):
        log.warning("map not found: %s (fallback to default)", fs_path)
        return f"/static/maps/{MAP_FILES['default']}"
    return f"/static/maps/{fname}"

def _first_ext_path(file_id: str) -> Optional[str]:
    pats = glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}.*"))
    return pats[0] if pats else None

def guess_floor_key(text: str | None) -> str:
    t = (text or "").lower().replace(" ", "")
    if ("2층" in t) or ("이층" in t) or ("2f" in t): return "2f"
    if ("1층" in t) or ("일층" in t) or ("1f" in t) or ("lobby" in t): return "1f"
    if ("b1" in t) or ("지하1층" in t): return "b1"
    return "default"

# ===============================
# 헬스체크
# ===============================
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

# ===============================
# 외부 호출
# ===============================
def run_stt(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        r = requests.post(f"{STT_SERVER_URL}/stt", files={"file": f}, timeout=60)
    r.raise_for_status()
    return (r.json() or {}).get("text", "")

def run_llm_intent(user_text: str) -> Dict[str, Any]:
    try:
        r = requests.post(f"{LLM_SERVER_URL}/intent", json={"text": user_text, "lang": "ko"}, timeout=20)
        if r.ok:
            return r.json()
    except Exception as e:
        log.warning("LLM intent error: %s (fallback used)", e)

    # ---- 폴백(rule-based) ----
    t = (user_text or "").lower()
    result = {"speak": "", "actions": [], "slots": {}}
    if any(k in t for k in ["지도", "맵", "map"]):
        fk = guess_floor_key(user_text)
        if fk == "default":
            # 층 정보가 없으면 층 선택 UI
            prompt = "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?"
            result["speak"] = prompt
            result["actions"].append({"service":"UI","name":"choose_floor",
                                      "params":{"floors": ["b1"] + [f"{i}f" for i in range(1,11)],
                                                "prompt": prompt}})
        else:
            result["speak"] = f"{'1층' if fk=='1f' else '2층' if fk=='2f' else '지하1층' if fk=='b1' else '지도를'} 보여드릴게요."
            result["slots"]["floor"] = fk
            result["actions"].append({"service":"INFO","name":"show_map","params":{"map_key": fk}})
    elif any(k in t for k in ["길 안내", "가는 길", "route", "네비"]):
        result["speak"] = "길 안내를 시작할게요."
        result["actions"].append({"service":"NAV","name":"plan_route","params":{"from":"로비","to":"전시실"}})
    elif any(k in t for k in ["직원", "도움", "호출", "call staff"]):
        result["speak"] = "직원을 호출할게요."
        result["actions"].append({"service":"CALL","name":"notify_staff","params":{}})
    elif any(k in t for k in ["영어", "english", "en "]):
        result["speak"] = "화면 언어를 영어로 바꿀게요."
        result["actions"].append({"service":"UI","name":"set_lang","params":{"lang":"en"}})
        result["slots"]["lang"] = "en"
    else:
        result["speak"] = "요청하신 정보를 찾고 있어요."
        result["actions"].append({"service":"INFO","name":"show_info","params":{"text":"준비 중입니다."}})
    return result

def tts_make_audio_url(text: str, speed: float = 1.0) -> str:
    """
    TTS 서버가 JSON으로 파일명/URL을 줄 때 최종 재생 URL을 만들어 반환.
    - {audio_url} 있으면 그대로 사용
    - {audio_path|filename} 이면 /ttsaudio/<basename> 으로 변환
    """
    try:
        r = requests.post(f"{TTS_SERVER_URL}/tts", json={"text": text, "speed": speed}, timeout=30)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        if data.get("audio_url"):
            return data["audio_url"]
        fname = data.get("audio_path") or data.get("filename")
        if fname:
            return f"/ttsaudio/{os.path.basename(fname)}"
    except Exception as e:
        log.warning("TTS error: %s", e)
    return ""

def run_notify_staff(payload: Optional[Dict[str, Any]] = None) -> bool:
    try:
        r = requests.post(f"{NOTIFY_SERVER_URL}/notify", json=payload or {"topic": "call_staff"}, timeout=5)
        return r.ok
    except Exception as e:
        log.warning("notify_staff failed: %s", e)
        return False

def run_nav_route(src: Optional[str], dst: Optional[str]) -> Dict[str, Any]:
    if dst and NAV_SERVER_URL:
        try:
            r = requests.post(f"{NAV_SERVER_URL}/route", json={"from": src, "to": dst}, timeout=5)
            if r.ok: return r.json()
        except Exception as e:
            log.warning("nav server failed, fallback: %s", e)
    steps = []
    if src: steps.append({"text": f"{src}에서 출발"})
    if dst:
        steps += [{"text": f"{dst}까지 직진 후 좌회전"}, {"text": "안내 표지판을 따라 50m 이동"}]
    return {"from": src, "to": dst, "steps": steps}

# ===============================
# 업로드/응답 API
# ===============================
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    프론트에서 업로드한 녹음 파일 저장.
    content-type 기준 확장자 부여(webm/ogg/wav/mp3)
    """
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
    url = await run_in_threadpool(tts_make_audio_url, text, speed)
    return JSONResponse({"audio_url": url})

# ===============================
# WebSocket: STT → LLM → ACTIONS → TTS
# ===============================
@app.websocket("/ws/kiosk")
async def kiosk_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # 하트비트
            if data.get("type") == "ping":
                continue

            # 선택된 층 처리(버튼 클릭)
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
                    tts_url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0)
                    if tts_url:
                        await ws.send_json({"stage":"tts","audio_url": tts_url})
                continue

            # 좌측 버튼의 가짜 유저 발화
            if data.get("type") == "synthetic_text":
                stt_text = (data.get("text") or "").strip()
                if not stt_text:
                    continue
                await ws.send_json({"stage":"stt","text": stt_text})
                await ws.send_json({"stage":"status","text":"의도 분석 중..."})
                try:
                    result = await run_in_threadpool(run_llm_intent, stt_text)
                    if not isinstance(result, dict): result = {}
                except Exception:
                    result = {}

                # 액션 처리
                for act in result.get("actions", []):
                    svc    = act.get("service")
                    name   = act.get("name")
                    params = (act.get("params") or {})

                    if svc == "INFO" and name in ("show_image","show_map","map"):
                        url = params.get("url")
                        if not url:
                            key = params.get("map_key") or result.get("slots", {}).get("floor")
                            if not key: key = guess_floor_key(stt_text)
                            url = resolve_map_url(key)
                        await ws.send_json({"stage":"ui","type":"show_map","url": url})

                    if svc == "INFO" and name in ("place_info","show_info"):
                        text = params.get("text") or result.get("speak") or ""
                        await ws.send_json({"stage":"ui","type":"show_info","html": text})

                    if svc == "NAV" and name == "plan_route":
                        route = await run_in_threadpool(run_nav_route, params.get("from"), params.get("to"))
                        await ws.send_json({"stage":"ui","type":"nav_route","route": route})

                    if svc == "CALL" and name in ("notify_staff","notify_security"):
                        ok = await run_in_threadpool(run_notify_staff, {"type":name, "meta":params})
                        await ws.send_json({"stage":"ui","type":"call_staff_ack","ok": bool(ok)})

                    if svc in ("UI","INFO") and name in ("set_lang","change_lang","set_language"):
                        lang = params.get("lang") or result.get("slots",{}).get("lang") or "ko"
                        await ws.send_json({"stage":"ui","type":"set_lang","lang": lang})

                speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
                await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})
                try:
                    url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0)
                    if url: await ws.send_json({"stage":"tts","audio_url": url})
                except Exception:
                    pass
                continue

            # ===== 파일 업로드 처리 이후 파이프라인 =====
            file_id = data.get("file_id") or (data.get("type") == "audio_uploaded" and data.get("file_id"))
            if not file_id:
                continue

            upath = _first_ext_path(file_id)
            if not upath:
                err_text = "업로드된 음성을 찾지 못했습니다. 다시 말씀해 주세요."
                await ws.send_json({"stage":"llm","text": err_text, "user_text": ""})
                url = await run_in_threadpool(tts_make_audio_url, err_text, 1.0)
                if url: await ws.send_json({"stage":"tts","audio_url": url})
                continue

            # ==== STT ====
            await ws.send_json({"stage":"status","text":"음성 인식 중..."})
            try:
                stt_text = await run_in_threadpool(run_stt, upath)
            except Exception:
                stt_text = ""

            if not stt_text:
                # 실패해도 반드시 TTS로 안내 → onended 후 재청취
                err_text = "음성 인식에 실패했어요. 다시 말씀해 주세요."
                await ws.send_json({"stage":"llm","text": err_text, "user_text": stt_text})
                url = await run_in_threadpool(tts_make_audio_url, err_text, 1.0)
                if url: await ws.send_json({"stage":"tts","audio_url": url})
                continue

            # STT 성공: 유저 말풍선 패킷
            await ws.send_json({"stage":"stt","text": stt_text})

            # ==== LLM ====
            await ws.send_json({"stage":"status","text":"의도 분석 중..."})
            try:
                result = await run_in_threadpool(run_llm_intent, stt_text)
                if not isinstance(result, dict): result = {}
            except Exception:
                result = {}

            # ==== 액션 처리 ====
            for act in result.get("actions", []):
                svc    = act.get("service")
                name   = act.get("name")
                params = (act.get("params") or {})

                if svc == "INFO" and name in ("show_image","show_map","map"):
                    url = params.get("url")
                    if not url:
                        key = params.get("map_key") or result.get("slots", {}).get("floor")
                        if not key: key = guess_floor_key(stt_text)   # 최후 폴백
                        url = resolve_map_url(key)
                    await ws.send_json({"stage":"ui","type":"show_map","url": url})

                if svc == "INFO" and name in ("place_info","show_info"):
                    text = params.get("text") or result.get("speak") or ""
                    await ws.send_json({"stage":"ui","type":"show_info","html": text})

                if svc == "NAV" and name == "plan_route":
                    route = await run_in_threadpool(run_nav_route, params.get("from"), params.get("to"))
                    await ws.send_json({"stage":"ui","type":"nav_route","route": route})

                if svc == "CALL" and name in ("notify_staff","notify_security"):
                    ok = await run_in_threadpool(run_notify_staff, {"type":name, "meta":params})
                    await ws.send_json({"stage":"ui","type":"call_staff_ack","ok": bool(ok)})

                if svc in ("UI","INFO") and name in ("set_lang","change_lang","set_language"):
                    lang = params.get("lang") or result.get("slots",{}).get("lang") or "ko"
                    await ws.send_json({"stage":"ui","type":"set_lang","lang": lang})

            # ==== 멘트 ====
            speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
            await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})

            # ==== TTS ====
            url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0)
            if url:
                await ws.send_json({"stage":"tts","audio_url": url})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception("WebSocket fatal: %s", e)
        await ws.close(code=1011)

# ===============================
# 프론트(정적) 서빙
# ===============================
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
