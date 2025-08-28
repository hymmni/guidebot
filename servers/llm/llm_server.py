# llm_server.py — Qwen 기반 인텐트 서버
# 실행:
#   conda activate llm-server
#   CUDA_VISIBLE_DEVICES=0 uvicorn llm_server:app --host 0.0.0.0 --port 9100

import os, re, json, time, logging, threading
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
MODEL_NAME  = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")  # VRAM 여유면 14B도 가능
DEVICE_MAP  = os.environ.get("LLM_DEVICE_MAP", "auto")
MAX_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "220"))

# ===============================
# 스키마
# ===============================
class IntentResult(BaseModel):
    intent: str = "UNKNOWN"
    slots: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    speak: str = "무엇을 도와드릴까요?"
    ask_user: str = ""
    confidence: float = 0.5
    _raw: Optional[str] = None

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
# 정규화/보강
# ===============================
def normalize_floor(val: Optional[str]) -> Optional[str]:
    if not val: return None
    t = re.sub(r"\s+", "", str(val).lower())
    table = {
        "1층":"1f","일층":"1f","lobby":"1f","1f":"1f",
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

def postprocess_result(r: Dict[str, Any], user_text: str, lang: str="ko") -> Dict[str, Any]:
    # 층 보강
    floor = (r.get("slots") or {}).get("floor")
    nf = normalize_floor(floor) or normalize_floor(user_text)
    if nf:
        r.setdefault("slots", {})["floor"] = nf

    # SHOW_MAP 보강
    if r.get("intent") == "SHOW_MAP":
        if not nf:
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
            has_map = any(a.get("service")=="INFO" and a.get("name") in ("show_map","show_image","map")
                          for a in r.get("actions", []))
            if not has_map:
                r.setdefault("actions", []).append({
                    "service":"INFO","name":"show_map","params":{"map_key": nf}
                })
            else:
                for a in r["actions"]:
                    if a.get("service")=="INFO" and a.get("name") in ("show_map","show_image","map"):
                        a.setdefault("params", {}).setdefault("map_key", nf)
            if not r.get("speak"):
                if nf.endswith("f") and nf[:-1].isdigit():
                    r["speak"] = f"{int(nf[:-1])}층 지도를 보여드릴게요."
                elif nf.startswith("b") and nf[1:].isdigit():
                    r["speak"] = f"지하 {int(nf[1:])}층 지도를 보여드릴게요."
                else:
                    r["speak"] = "지도를 보여드릴게요."
    return r

# ===============================
# 시스템 프롬프트
# ===============================
SYSTEM_PROMPT = """You are an intent parser for a kiosk in Korean named "에스뽀".
Return ONLY strict JSON with keys: intent, slots, actions, speak, ask_user, confidence.
- intent ∈ [SHOW_MAP, ROUTE_REQUEST, PLACE_INFO, CALL_STAFF, TRANSLATE, SET_LANGUAGE, SMALL_TALK, ALARM_EMERGENCY, UNKNOWN]
- slots may include: floor ('1f','2f','b1'...), destination/to, from, lang ('ko','en','ja','zh'...)
- actions: array of {service, name, params}. For SHOW_MAP include params.map_key if possible.
- speak: short, kind, and cheerful reply (<=40 words) in the SAME LANGUAGE as the user's utterance.
- ask_user: if info is missing, one cute and friendly clarifying question in the user's language; else empty.
- confidence: 0..1 number.

Rules:
- Your name is "에스뽀". If user uses another language, answer in that language.
- If asked "show map" without a floor, DO NOT guess. Ask which floor (B1~10F) and propose a floor choice UI.
- Your personality is like a very friendly, cute, and helpful friend! Please end your sentences with cute endings like `~용`, `~죠`, `~뿅!`, or `~이요`. Use emojis often!
- Output JSON only, no code fences.
"""

FEW_SHOTS = [
    ('"1층 지도 보여줘"', "ko", {
        "intent":"SHOW_MAP",
        "slots":{"floor":"1f"},
        "speak":"1층 지도를 보여드릴게요.",
        "ask_user":"",
        "confidence":0.9,
        "actions":[{"service":"INFO","name":"show_map","params":{"map_key":"1f"}}]
    }),
    ('"지도"', "ko", {
        "intent":"SHOW_MAP",
        "slots":{},
        "speak":"지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?",
        "ask_user":"지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?",
        "confidence":0.8,
        "actions":[{"service":"UI","name":"choose_floor","params":{"floors":["b1"]+[f"{i}f" for i in range(1,11)],"prompt":"지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?"}}]
    }),
    ('"영어로 바꿔줘"', "ko", {
        "intent":"SET_LANGUAGE",
        "slots":{"lang":"en"},
        "speak":"화면 언어를 영어로 바꿀게요.",
        "ask_user":"",
        "confidence":0.95,
        "actions":[{"service":"UI","name":"set_lang","params":{"lang":"en"}}]
    }),
]

def build_messages(user: str) -> List[Dict[str,str]]:
    msgs=[{"role":"system","content": SYSTEM_PROMPT}]
    for q,ql,outj in FEW_SHOTS:
        msgs.append({"role":"user","content": f"[lang={ql}] {q}"})
        msgs.append({"role":"assistant","content": json.dumps(outj, ensure_ascii=False)})
    msgs.append({"role":"user","content": user})
    return msgs

# ===============================
# 모델 로딩
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

# ── 금관 콘텐츠(LLM이 내려줌: title/url/desc + 긴설명 TTS 액션)
GEUMGWAN = {
    "title": "금관",
    "url": "/artifacts/geumgwan.png",
    "desc": (
        "이 금관은 신라의 왕권과 종교적 상징을 함께 담은 대표 유물입니다. "
        "사슴뿔 모양의 가지 장식과 나뭇가지 모양의 수식이 위로 뻗어 하늘과의 연결을 표현하고, "
        "얇은 금판을 오려 만든 세공과 옥으로 장식된 곁가지가 섬세합니다. "
        "착용은 일상용이라기보다 의례나 장송 의식에서 사용된 것으로 추정됩니다. "
        "순금판을 얇게 가공해 무게를 줄였고, 금실과 구슬 장식은 움직일 때마다 빛과 소리를 냈습니다. "
        "정치·종교·예술이 결합된 신라 문화의 정수를 보여줍니다."
    )
}

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
    text = (data.get("text") or "").strip()
    sid  = data.get("session_id","default")

    # 1) LLM 호출
    messages = build_messages(text)
    prompt   = apply_chat_template(messages)
    raw      = call_llm(prompt)

    # 2) 1차 파싱
    try:
        block = extract_json_block(raw)
        obj   = json.loads(block)
        res   = coerce_result(obj)
        res._raw = raw
    except Exception as e1:
        # 3) 복구 프롬프트 (간단)
        repair_msgs = [
            {"role":"system","content":"Fix to valid JSON. Output JSON ONLY. Keys: intent, slots, actions, speak, ask_user, confidence"},
            {"role":"user","content": raw}
        ]
        raw2 = call_llm(apply_chat_template(repair_msgs))
        try:
            block = extract_json_block(raw2)
            obj   = json.loads(block)
            res   = coerce_result(obj)
            res._raw = raw
        except Exception as e2:
            log.error(f"[parse fail] {e1} / {e2}")
            res = IntentResult()

    # 4) 후처리 보강(층 선택, map_key 등)
    res_dict = res.model_dump()
    res_dict = postprocess_result(res_dict, text, "ko")

    # 5) 금관 시나리오(규칙 보강): “정보/설명” → 선택 패널, “금관” → 이미지+긴 설명 TTS
    t = text.lower()
    if any(k in t for k in ["정보","설명","안내","explain","info"]) and not any(a.get("name")=="choice" for a in res_dict.get("actions",[])):
        res_dict.setdefault("actions", []).append({
            "service":"UI","name":"choice",
            "params":{
                "title":"정보",
                "prompt":"무엇을 설명해드릴까요?",
                "options":[{"label":"금관","say":"금관 설명해줘"}]
            }
        })
        res_dict["speak"] = res_dict.get("speak") or "무엇을 설명해드릴까요?"

    if any(k in t for k in ["금관","geumgwan","gold crown","金冠"]):
        if not any(a.get("name")=="show_artifact" for a in res_dict.get("actions",[])):
            res_dict.setdefault("actions", []).append({
                "service":"INFO","name":"show_artifact",
                "params":{"title":GEUMGWAN["title"],"url":GEUMGWAN["url"],"desc":GEUMGWAN["desc"]}
            })
        if not any(a.get("name")=="speak_detail" for a in res_dict.get("actions",[])):
            res_dict.setdefault("actions", []).append({
                "service":"TTS","name":"speak_detail","params":{"text": GEUMGWAN["desc"]}
            })
        res_dict["speak"] = res_dict.get("speak") or "금관에 대해 알려드릴게요."

    # 6) 메모리 저장
    MEM.put(sid, text, IntentResult(**res_dict))
    return JSONResponse(res_dict)

@app.post("/reset")
async def reset_session(req: Request):
    data = await req.json()
    sid = data.get("session_id","default")
    MEM.reset(sid)
    return {"status":"ok","session_id":sid}
