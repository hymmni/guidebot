# llm_server.py — Qwen 로컬 LLM 인텐트 서버 (FastAPI)
# 실행 예:
#   conda activate llm-server
#   CUDA_VISIBLE_DEVICES=0 uvicorn llm_server:app --host 0.0.0.0 --port 9100

import os, re, json, time, logging, threading
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("llm_server")

# ===============================
# 설정
# ===============================
MODEL_NAME  = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")  # VRAM 부족시 7B로
DEVICE_MAP  = os.environ.get("LLM_DEVICE_MAP", "auto")                   # "auto" 권장
MAX_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "220"))

# ===============================
# 스키마
# ===============================
class Intent(str, Enum):
    ROUTE_REQUEST = "ROUTE_REQUEST"
    PLACE_INFO    = "PLACE_INFO"
    CALL_STAFF    = "CALL_STAFF"
    TRANSLATE     = "TRANSLATE"
    SET_LANGUAGE  = "SET_LANGUAGE"
    SHOW_MAP      = "SHOW_MAP"
    SMALL_TALK    = "SMALL_TALK"
    ALARM_EMERGENCY = "ALARM_EMERGENCY"
    UNKNOWN       = "UNKNOWN"

class IntentResult(BaseModel):
    intent: Intent = Intent.UNKNOWN
    slots: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    speak: str = "무엇을 도와드릴까요?"
    ask_user: str = ""
    confidence: float = 0.5
    _raw: Optional[str] = None

class IntentReq(BaseModel):
    text: str
    lang: str = "ko"
    session_id: str = "default"

# ===============================
# 간단 세션 메모리
# ===============================
@dataclass
class Turn:
    user: str
    result: IntentResult
    ts: float

class Memory:
    def __init__(self, max_turns=8, ttl_sec=3600):
        self.max_turns = max_turns
        self.ttl = ttl_sec
        self.lock = threading.Lock()
        self.store: Dict[str, List[Turn]] = {}
    def put(self, sid: str, user: str, result: IntentResult):
        with self.lock:
            arr = self.store.setdefault(sid, [])
            arr.append(Turn(user=user, result=result, ts=time.time()))
            if len(arr) > self.max_turns:
                self.store[sid] = arr[-self.max_turns:]
    def get_context(self, sid: str) -> List[Turn]:
        now = time.time()
        with self.lock:
            arr = [t for t in self.store.get(sid, []) if now - t.ts < self.ttl]
            self.store[sid] = arr
            return arr
    def reset(self, sid: str):
        with self.lock:
            self.store.pop(sid, None)

MEM = Memory()

# ===============================
# 정규화/후처리
# ===============================
def normalize_floor(val: Optional[str]) -> Optional[str]:
    if not val: return None
    t = re.sub(r"\s+", "", str(val).lower())
    table = {
        "1층":"1f","일층":"1f","lobby":"1f","ground":"1f","1f":"1f",
        "2층":"2f","이층":"2f","2f":"2f",
        "지하1층":"b1","b1":"b1",
    }
    for k,v in table.items():
        if k in t: return v
    m = re.search(r"(지하|b)\s*(\d+)", t)
    if m: return f"b{m.group(2)}"
    m = re.search(r"(\d+)\s*(층|f)", t)
    if m: return f"{int(m.group(1))}f"
    return None

# llm_server.py 中 (기존 postprocess_result를 아래로 교체)

def postprocess_result(r: Dict[str, Any], user_text: str, lang: str="ko") -> Dict[str, Any]:
    from pydantic import ValidationError
    try:
        # 기존 IntentResult 검증 코드가 있다면 그대로 유지
        pass
    except ValidationError:
        pass

    # 층 정규화 유틸 (이미 있다면 재사용)
    def normalize_floor(val: Optional[str]) -> Optional[str]:
        import re
        if not val: return None
        t = re.sub(r"\s+", "", str(val).lower())
        tbl = {"1층":"1f","일층":"1f","lobby":"1f","ground":"1f","1f":"1f",
               "2층":"2f","이층":"2f","2f":"2f",
               "지하1층":"b1","b1":"b1"}
        for k,v in tbl.items():
            if k in t: return v
        m = re.search(r"(지하|b)\s*(\d+)", t)
        if m: return f"b{m.group(2)}"
        m = re.search(r"(\d+)\s*(층|f)", t)
        if m: return f"{int(m.group(1))}f"
        return None

    # 표준화
    floor = (r.get("slots") or {}).get("floor")
    floor = normalize_floor(floor) or normalize_floor(user_text)
    if floor:
        r.setdefault("slots", {})["floor"] = floor

    # ★ SHOW_MAP 처리
    if r.get("intent") == "SHOW_MAP":
        # 층이 없는 경우 → 층 선택 UI + 질문 멘트
        if not floor:
            prompt = "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?"
            r["speak"] = r.get("speak") or prompt
            r["ask_user"] = r.get("ask_user") or prompt
            r.setdefault("actions", []).append({
                "service": "UI",
                "name": "choose_floor",
                "params": {
                    "floors": ["b1"] + [f"{i}f" for i in range(1, 11)],
                    "prompt": prompt
                }
            })
        else:
            # 층이 있으면 show_map 보장
            has_map = any(a.get("service")=="INFO" and a.get("name") in ("show_map","show_image","map")
                          for a in r.get("actions", []))
            if not has_map:
                r.setdefault("actions", []).append({
                    "service":"INFO","name":"show_map","params":{"map_key": floor}
                })
            else:
                for a in r["actions"]:
                    if a.get("service")=="INFO" and a.get("name") in ("show_map","show_image","map"):
                        a.setdefault("params", {}).setdefault("map_key", floor)
            if not r.get("speak"):
                if floor.endswith("f") and floor[:-1].isdigit():
                    r["speak"] = f"{int(floor[:-1])}층 지도를 보여드릴게요."
                elif floor.startswith("b") and floor[1:].isdigit():
                    r["speak"] = f"지하 {int(floor[1:])}층 지도를 보여드릴게요."
                else:
                    r["speak"] = "지도를 보여드릴게요."

    # 나머지 인텐트 보강(필요 시 기존 로직 유지)
    return r

# ===============================
# 메시지/프롬프트
# ===============================
SYSTEM_PROMPT = """You are an intent parser for a kiosk in Korean named "에스뽀" (Esppo).
Return ONLY strict JSON with keys: intent, slots, actions, speak, ask_user, confidence.
- intent ∈ [SHOW_MAP, ROUTE_REQUEST, PLACE_INFO, CALL_STAFF, TRANSLATE, SET_LANGUAGE, SMALL_TALK, ALARM_EMERGENCY, UNKNOWN]
- slots may include: floor ('1f','2f','b1'...), destination/to, from, lang ('ko','en','ja','zh','es','fr','vi','th'...)
- actions: array of {service, name, params}. For SHOW_MAP include params.map_key normalized like floor if possible.
- speak: short reply (<=20 words) in the SAME LANGUAGE as the user's utterance.
- ask_user: if info is missing, one clarifying question in the user's language; else empty string.
- confidence: 0..1 number.

Name & language rules:
- Your name is "에스뽀". If the user greets or calls your name, treat it as SMALL_TALK and reply briefly in the user's language.
- Auto-detect the user's language from the utterance. If the user speaks in a foreign language, reply in that language and set slots.lang with an ISO-like code (ko, en, ja, zh, etc.).
- If the user explicitly asks to change the UI language, intent=SET_LANGUAGE and normalize slots.lang accordingly (ko, en, ja, zh, es, fr, vi, th).

Disambiguation examples (guidance):
- If asked to "show map" without a floor, DO NOT guess a floor. Put ask_user like "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?" and omit actions or provide SHOW_MAP without map_key until clarified.
- Keep all texts concise and polite.

Output JSON only, no code fences.
"""

FEW_SHOTS = [
    ('"1층 지도 보여줘"', "ko", {
        "intent":"SHOW_MAP",
        "slots":{"floor":"1f"},
        "speak":"1층 지도를 보여드릴게요.",
        "ask_user":"",
        "confidence":0.85,
        "actions":[{"service":"INFO","name":"show_map","params":{"map_key":"1f"}}]
    }),
    ('"2층 안내"', "ko", {
        "intent":"SHOW_MAP",
        "slots":{"floor":"2f"},
        "speak":"2층 지도를 보여드릴게요.",
        "ask_user":"",
        "confidence":0.85,
        "actions":[{"service":"INFO","name":"show_map","params":{"map_key":"2f"}}]
    }),
    ('"영어로 바꿔줘"', "ko", {
        "intent":"SET_LANGUAGE",
        "slots":{"lang":"en"},
        "speak":"화면 언어를 영어로 바꿀게요.",
        "ask_user":"",
        "confidence":0.9,
        "actions":[{"service":"UI","name":"set_lang","params":{"lang":"en"}}]
    }),
]

def build_messages(user: str, lang: str, ctx: List[Turn]) -> List[Dict[str,str]]:
    # 최근 문맥 요약(필요하면 사용)
    ctx_lines=[]
    for t in ctx[-4:]:
        try:
            ctx_lines.append(f"- intent={t.result.intent}, slots={json.dumps(t.result.slots, ensure_ascii=False)}")
        except: pass
    ctx_text="\n".join(ctx_lines) if ctx_lines else "(none)"

    msgs=[{"role":"system","content": SYSTEM_PROMPT + f"\n[State] ui_lang={lang}\n[RecentTurns]\n{ctx_text}\n[Output] JSON only."}]
    for q,ql,outj in FEW_SHOTS:
        msgs.append({"role":"user","content": f"[lang={ql}] {q}"})
        msgs.append({"role":"assistant","content": json.dumps(outj, ensure_ascii=False)})
    msgs.append({"role":"user","content": f"[lang={lang}] \"{user}\""})
    return msgs

# ===============================
# Qwen 로딩/호출
# ===============================
log.info(f"Loading model: {MODEL_NAME} (device_map={DEVICE_MAP})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
gen = pipeline(
    task="text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=DEVICE_MAP,
)
gen.model.eval()
torch.set_grad_enabled(False)

def apply_chat_template(messages: List[Dict[str,str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # 일반 모델 대비 폴백
        txt=""
        for m in messages:
            txt += f"<|{m['role']}|>\n{m['content'].strip()}\n"
        txt += "<|assistant|>\n"
        return txt

def call_llm(prompt: str) -> str:
    terminators=[]
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != tokenizer.eos_token_id:
            terminators.append(im_end)
    except Exception:
        pass

    out = gen(prompt, max_new_tokens=MAX_TOKENS, do_sample=False, temperature=0.0, top_p=1.0,
              eos_token_id=terminators if terminators else None)
    text = out[0]["generated_text"]
    if text.startswith(prompt): text = text[len(prompt):]
    return text.strip()

# ===============================
# JSON 파싱 유틸
# ===============================
def extract_json_block(s: str) -> str:
    if not s: return s
    s = s.replace("```json","").replace("```","").strip()
    start = s.find("{")
    if start < 0: return s
    depth=0; in_str=False; esc=False
    for i in range(start, len(s)):
        ch=s[i]
        if in_str:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=='"': in_str=False
        else:
            if ch=='"': in_str=True
            elif ch=='{': depth+=1
            elif ch=='}':
                depth-=1
                if depth==0: return s[start:i+1]
    return s[start:]

def coerce_result(obj: Dict[str, Any]) -> IntentResult:
    obj.setdefault("intent", "UNKNOWN")
    obj.setdefault("slots", {})
    obj.setdefault("actions", [])
    obj.setdefault("speak", "무엇을 도와드릴까요?")
    obj.setdefault("ask_user", "")
    obj.setdefault("confidence", 0.5)
    if obj["slots"] is None: obj["slots"] = {}
    if obj["actions"] is None: obj["actions"] = []
    try:
        return IntentResult(**obj)
    except ValidationError as e:
        log.warning(f"coerce_result validation fail: {e}")
        return IntentResult()

def run_intent_llm(user: str, lang: str, ctx: List[Turn]) -> IntentResult:
    messages = build_messages(user, lang, ctx)
    prompt = apply_chat_template(messages)
    raw = call_llm(prompt)
    log.info(f"[LLM raw] {raw[:400]}{'...' if len(raw)>400 else ''}")

    # 1차 파싱
    try:
        block = extract_json_block(raw)
        data = json.loads(block)
        res = coerce_result(data)
        res._raw = raw
        return res
    except Exception as e1:
        log.warning(f"[1st parse fail] {e1}")

    # 2차: 복구 프롬프트
    repair_msgs = [
        {"role":"system","content":"Fix to valid JSON. Output JSON ONLY. Keys: intent, slots, actions, speak, ask_user, confidence"},
        {"role":"user","content": raw}
    ]
    raw2 = call_llm(apply_chat_template(repair_msgs))
    try:
        block = extract_json_block(raw2)
        data = json.loads(block)
        res = coerce_result(data)
        res._raw = raw
        return res
    except Exception as e2:
        log.error(f"[2nd parse fail] {e2}")
        return IntentResult(_raw=raw)

# ===============================
# FastAPI
# ===============================
app = FastAPI(title="Qwen Intent Server", version="1.0")

@app.get("/health")
def health():
    try:
        _ = gen("ping", max_new_tokens=1, do_sample=False)
        return {"status":"ok"}
    except Exception as e:
        return JSONResponse({"status":"error","detail":str(e)}, status_code=503)

@app.post("/intent")
async def intent(req: Request):
    data = await req.json()
    text = data.get("text","")
    lang = data.get("lang","ko")
    sid  = data.get("session_id","default")

    ctx = MEM.get_context(sid)
    res = run_intent_llm(text, lang, ctx)
    res_dict = res.model_dump()
    res_dict = postprocess_result(res_dict, text, lang)   # ★ 층수/액션 보강
    MEM.put(sid, text, IntentResult(**res_dict))
    return JSONResponse(res_dict)

@app.post("/reset")
async def reset_session(req: Request):
    data = await req.json()
    sid = data.get("session_id","default")
    MEM.reset(sid)
    return {"status":"ok","session_id":sid}
