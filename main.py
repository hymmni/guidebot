# main.py
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── NEW: 유물 이미지 서빙 (/artifacts)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
if os.path.isdir(ARTIFACT_DIR):
    app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR), name="artifacts")
else:
    log.warning("ARTIFACT_DIR not found: %s", ARTIFACT_DIR)

UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", "/tmp/audio_kiosk")
TTS_OUT_DIR = os.path.join(BASE_DIR, "melotts_output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_OUT_DIR, exist_ok=True)
app.mount("/ttsaudio", StaticFiles(directory=TTS_OUT_DIR), name="ttsaudio")

STT_SERVER_URL    = os.environ.get("STT_URL",   "http://localhost:9000")
LLM_SERVER_URL    = os.environ.get("LLM_URL",   "http://localhost:9100")
TTS_SERVER_URL    = os.environ.get("TTS_URL",   "http://localhost:9200")
NOTIFY_SERVER_URL = os.environ.get("NOTIFY_URL","http://localhost:9300")
NAV_SERVER_URL    = os.environ.get("NAV_URL",   "")

MAP_FILES = {
    "default": "venue_map.png",
    "1f": "floor1.png",
    "2f": "floor2.png",
    "b1": "b1.png",
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

def guess_floor_key(text: str | None) -> str:
    t = (text or "").lower().replace(" ", "")
    if ("2층" in t) or ("이층" in t) or ("2f" in t): return "2f"
    if ("1층" in t) or ("일층" in t) or ("1f" in t) or ("lobby" in t): return "1f"
    if ("b1" in t) or ("지하1층" in t): return "b1"
    return "default"

# ── NEW: 금관 설명/이미지 리소스
def get_geumgwan_payload() -> Dict[str, Any]:
    url = "/artifacts/geumgwan.png"
    # 필요하면 파일 존재 체크
    if not os.path.exists(os.path.join(ARTIFACT_DIR, "geumgwan.png")):
        log.warning("geumgwan.png not found under ./artifacts/")
    desc = (
        "이 금관은 신라의 왕권과 종교적 상징을 함께 담은 대표 유물입니다. "
        "사슴뿔 모양의 가지 장식과 나뭇가지 모양의 수식이 위로 뻗어 하늘과의 연결을 표현하고, "
        "얇은 금판을 오려 만든 세공과 옥으로 장식된 곁가지가 섬세합니다. "
        "착용 시에는 실제 왕이 쓰기보다는 의례나 장송 의식에서 사용한 것으로 추정됩니다. "
        "무게를 줄이기 위해 순금판을 얇게 가공했고, 금실과 구슬 장식이 움직일 때마다 빛과 소리를 냈습니다. "
        "발견된 고분의 부장품 구성과 함께 볼 때, 당시 정치·종교·예술이 결합된 신라 문화의 정수를 보여줍니다."
    )
    speak = "신라 금관은 왕권과 종교적 상징을 담은 대표 유물로, 사슴뿔 형태의 장식과 섬세한 금세공이 특징입니다."
    return {
        "title": "금관",
        "url": url,
        "desc": desc,
        "speak": speak,  # 짧은 멘트(채팅에 표시). 실제 TTS는 desc를 길게 읽게 할 예정
    }

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

def run_stt(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        r = requests.post(f"{STT_SERVER_URL}/stt", files={"file": f}, timeout=60)
    r.raise_for_status()
    return (r.json() or {}).get("text", "")

def run_llm_intent(user_text: str) -> Dict[str, Any]:
    # 우선 LLM 시도
    try:
        r = requests.post(f"{LLM_SERVER_URL}/intent", json={"text": user_text, "lang": "ko"}, timeout=20)
        if r.ok:
            return r.json()
    except Exception as e:
        log.warning("LLM intent error: %s (fallback used)", e)

    # ── Fallback(rule-based)
    t = (user_text or "").lower()
    result = {"speak": "", "actions": [], "slots": {}}

    # 금관 직접 요청
    if ("금관" in t) or ("geumgwan" in t) or ("금 왕관" in t):
        art = get_geumgwan_payload()
        result["speak"] = art["speak"]
        result["actions"].append({"service":"INFO","name":"show_artifact","params":{"title": art["title"], "url": art["url"], "desc": art["desc"]}})
        # 길게 읽을 텍스트를 별도로 전달
        result["actions"].append({"service":"TTS","name":"speak_detail","params":{"text": art["desc"]}})
        return result

    # 정보/설명 → 선택 패널 유도
    if any(k in t for k in ["정보", "설명", "안내", "explain", "info"]):
        prompt = "무엇을 설명해드릴까요?"
        result["speak"] = prompt
        result["actions"].append({
            "service":"UI","name":"choice",
            "params":{
                "title":"정보",
                "prompt":prompt,
                "options":[
                    {"label":"금관", "say":"금관 설명해줘"}
                ]
            }
        })
        return result

    # 지도
    if any(k in t for k in ["지도", "맵", "map"]):
        fk = guess_floor_key(user_text)
        if fk == "default":
            prompt = "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?"
            result["speak"] = prompt
            result["actions"].append({"service":"UI","name":"choose_floor",
                                      "params":{"floors": ["b1"] + [f"{i}f" for i in range(1,11)],
                                                "prompt": prompt}})
        else:
            result["speak"] = f"{'1층' if fk=='1f' else '2층' if fk=='2f' else '지하1층' if fk=='b1' else '지도를'} 보여드릴게요."
            result["slots"]["floor"] = fk
            result["actions"].append({"service":"INFO","name":"show_map","params":{"map_key": fk}})
        return result

    # 길 안내
    if any(k in t for k in ["길 안내","가는 길","route","네비"]):
        result["speak"] = "길 안내를 시작할게요."
        result["actions"].append({"service":"NAV","name":"plan_route","params":{"from":"로비","to":"전시실"}})
        return result

    # 직원 호출
    if any(k in t for k in ["직원","도움","호출","call staff"]):
        result["speak"] = "직원을 호출할게요."
        result["actions"].append({"service":"CALL","name":"notify_staff","params":{}})
        return result

    # 언어 변경
    if any(k in t for k in ["영어","english","en "]):
        result["speak"] = "화면 언어를 영어로 바꿀게요."
        result["actions"].append({"service":"UI","name":"set_lang","params":{"lang":"en"}})
        result["slots"]["lang"] = "en"
        return result

    result["speak"] = "요청하신 정보를 찾고 있어요."
    result["actions"].append({"service":"INFO","name":"show_info","params":{"text":"준비 중입니다."}})
    return result

def tts_make_audio_url(text: str, speed: float = 1.0) -> str:
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
    url = await run_in_threadpool(tts_make_audio_url, text, speed)
    return JSONResponse({"audio_url": url})

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

            if data.get("type") == "ping":
                continue

            # ── 층 선택 버튼 처리
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

            # ── 좌측 버튼(가짜 유저 발화)
            if data.get("type") == "synthetic_text":
                stt_text = (data.get("text") or "").strip()
                if not stt_text: continue
                await ws.send_json({"stage":"stt","text": stt_text})
                await ws.send_json({"stage":"status","text":"의도 분석 중..."})
                try:
                    result = await run_in_threadpool(run_llm_intent, stt_text)
                    if not isinstance(result, dict): result = {}
                except Exception:
                    result = {}

                # 액션
                for act in result.get("actions", []):
                    svc    = act.get("service")
                    name   = act.get("name")
                    params = (act.get("params") or {})

                    if svc == "UI" and name in ("choice",):
                        await ws.send_json({"stage":"ui","type":"choice","title": params.get("title",""),"prompt":params.get("prompt",""),"options": params.get("options",[])})
                    if svc == "INFO" and name in ("show_image","show_map","map"):
                        url = params.get("url") or resolve_map_url(params.get("map_key"))
                        await ws.send_json({"stage":"ui","type":"show_map","url": url})
                    if svc == "INFO" and name in ("place_info","show_info"):
                        text = params.get("text") or result.get("speak") or ""
                        await ws.send_json({"stage":"ui","type":"show_info","html": text})
                    # ── NEW: 유물
                    if svc == "INFO" and name == "show_artifact":
                        await ws.send_json({"stage":"ui","type":"show_artifact","title": params.get("title",""), "url": params.get("url",""), "desc": params.get("desc","")})
                    if svc == "NAV" and name == "plan_route":
                        route = await run_in_threadpool(run_nav_route, params.get("from"), params.get("to"))
                        await ws.send_json({"stage":"ui","type":"nav_route","route": route})
                    if svc == "CALL" and name in ("notify_staff","notify_security"):
                        ok = await run_in_threadpool(run_notify_staff, {"type":name, "meta":params})
                        await ws.send_json({"stage":"ui","type":"call_staff_ack","ok": bool(ok)})
                    if svc in ("UI","INFO") and name in ("set_lang","change_lang","set_language"):
                        lang = params.get("lang") or result.get("slots",{}).get("lang") or "ko"
                        await ws.send_json({"stage":"ui","type":"set_lang","lang": lang})
                    # ── NEW: 긴 설명을 별도 TTS로
                    if svc == "TTS" and name == "speak_detail":
                        long_txt = (params.get("text") or "").strip()
                        if long_txt:
                            url = await run_in_threadpool(tts_make_audio_url, long_txt, 1.0)
                            if url: await ws.send_json({"stage":"tts","audio_url": url})

                speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
                await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})
                try:
                    url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0)
                    if url: await ws.send_json({"stage":"tts","audio_url": url})
                except Exception:
                    pass
                continue

            # ── 파일 업로드 파이프라인
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

            await ws.send_json({"stage":"status","text":"음성 인식 중..."})
            try:
                stt_text = await run_in_threadpool(run_stt, upath)
            except Exception:
                stt_text = ""

            if not stt_text:
                err_text = "음성 인식에 실패했어요. 다시 말씀해 주세요."
                await ws.send_json({"stage":"llm","text": err_text, "user_text": stt_text})
                url = await run_in_threadpool(tts_make_audio_url, err_text, 1.0)
                if url: await ws.send_json({"stage":"tts","audio_url": url})
                continue

            await ws.send_json({"stage":"stt","text": stt_text})
            await ws.send_json({"stage":"status","text":"의도 분석 중..."})
            try:
                result = await run_in_threadpool(run_llm_intent, stt_text)
                if not isinstance(result, dict): result = {}
            except Exception:
                result = {}

            for act in result.get("actions", []):
                svc    = act.get("service")
                name   = act.get("name")
                params = (act.get("params") or {})

                if svc == "UI" and name in ("choice",):
                    await ws.send_json({"stage":"ui","type":"choice","title": params.get("title",""),"prompt":params.get("prompt",""),"options": params.get("options",[])})
                if svc == "INFO" and name in ("show_image","show_map","map"):
                    url = params.get("url")
                    if not url:
                        key = params.get("map_key") or guess_floor_key(stt_text)
                        url = resolve_map_url(key)
                    await ws.send_json({"stage":"ui","type":"show_map","url": url})
                if svc == "INFO" and name in ("place_info","show_info"):
                    text = params.get("text") or result.get("speak") or ""
                    await ws.send_json({"stage":"ui","type":"show_info","html": text})
                if svc == "INFO" and name == "show_artifact":
                    await ws.send_json({"stage":"ui","type":"show_artifact","title": params.get("title",""), "url": params.get("url",""), "desc": params.get("desc","")})
                if svc == "NAV" and name == "plan_route":
                    route = await run_in_threadpool(run_nav_route, params.get("from"), params.get("to"))
                    await ws.send_json({"stage":"ui","type":"nav_route","route": route})
                if svc == "CALL" and name in ("notify_staff","notify_security"):
                    ok = await run_in_threadpool(run_notify_staff, {"type":name, "meta":params})
                    await ws.send_json({"stage":"ui","type":"call_staff_ack","ok": bool(ok)})
                if svc in ("UI","INFO") and name in ("set_lang","change_lang","set_language"):
                    lang = params.get("lang") or "ko"
                    await ws.send_json({"stage":"ui","type":"set_lang","lang": lang})
                if svc == "TTS" and name == "speak_detail":
                    long_txt = (params.get("text") or "").strip()
                    if long_txt:
                        url = await run_in_threadpool(tts_make_audio_url, long_txt, 1.0)
                        if url: await ws.send_json({"stage":"tts","audio_url": url})

            speak_text = result.get("speak") or result.get("reply") or "알겠습니다."
            await ws.send_json({"stage":"llm","text": speak_text, "user_text": stt_text})
            url = await run_in_threadpool(tts_make_audio_url, speak_text, 1.0)
            if url:
                await ws.send_json({"stage":"tts","audio_url": url})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception("WebSocket fatal: %s", e)
        await ws.close(code=1011)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
