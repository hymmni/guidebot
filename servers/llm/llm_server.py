# llm_server.py
# FastAPI 기반 관광안내 로봇 LLM 서버 (문맥 유지 + 안전 파싱 + 추천 후처리)
# ---------------------------------------------------------------
# 실행:
#   conda activate llm-server
#   CUDA_VISIBLE_DEVICES=0 uvicorn llm_server:app --host 0.0.0.0 --port 9100
# ---------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, logging, threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, pipeline

# --------------------------
# 설정 & 로깅
# --------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("llm_server")

MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
DEVICE_MAP = os.environ.get("LLM_DEVICE_MAP", "auto")  # "auto" 권장

# --------------------------
# 인텐트 / 스키마
# --------------------------
class Intent(str, Enum):
    ROUTE_REQUEST = "ROUTE_REQUEST"   # 길안내(출발/도착)
    PLACE_INFO    = "PLACE_INFO"      # 시설/장소 정보
    CALL_STAFF    = "CALL_STAFF"      # 직원 호출/도움요청
    TRANSLATE     = "TRANSLATE"       # 번역
    SET_LANGUAGE  = "SET_LANGUAGE"    # 언어 설정
    SMALL_TALK    = "SMALL_TALK"      # 잡담/안부
    ALARM_EMERGENCY = "ALARM_EMERGENCY" # 비상/신고
    UNKNOWN       = "UNKNOWN"         # 분류 불가

class IntentResult(BaseModel):
    intent: Intent = Intent.UNKNOWN
    slots: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    speak: str = "무엇을 도와드릴까요?"
    ask_user: str = ""
    confidence: float = 0.5
    # 디버깅용(클라이언트가 원하지 않으면 무시 가능)
    _raw: Optional[str] = None

class IntentReq(BaseModel):
    text: str
    lang: str = "ko"
    session_id: str = "default"   # 프론트에서 연결별 고유값 전달 권장

# --------------------------
# 경량 문맥 메모리(세션별)
# --------------------------
@dataclass
class Turn:
    user: str
    result: IntentResult
    ts: float

class Memory:
    """세션별 최근 대화(최대 N턴) + TTL 관리"""
    def __init__(self, max_turns=10, ttl_sec=3600):
        self.max_turns = max_turns
        self.ttl = ttl_sec
        self.lock = threading.Lock()
        self.store: Dict[str, List[Turn]] = {}

    def put(self, sid: str, user: str, result: IntentResult):
        with self.lock:
            turns = self.store.setdefault(sid, [])
            turns.append(Turn(user=user, result=result, ts=time.time()))
            if len(turns) > self.max_turns:
                self.store[sid] = turns[-self.max_turns:]

    def get_context(self, sid: str) -> List[Turn]:
        now = time.time()
        with self.lock:
            turns = self.store.get(sid, [])
            # TTL 정리
            turns = [t for t in turns if now - t.ts < self.ttl]
            self.store[sid] = turns
            return turns

    def reset(self, sid: str):
        with self.lock:
            self.store.pop(sid, None)

MEM = Memory(max_turns=8, ttl_sec=3600)

# --------------------------
# 경량 지식/추천 룰 (예시)
# --------------------------
# STT 오인식 보정(간단 사전)
NORMALIZE_MAP = {
    "상실": "화장실",
    "화잘실": "화장실",
    "홧장실": "화장실",
}

# 장소 동의어 → 표준 슬롯값
PLACE_ALIASES = {
    "화장실": {"화장실", "restroom", "toilet", "washroom", "wc", "상실"},
    "안내데스크": {"안내", "안내소", "information desk", "데스크"},
    "카페": {"카페", "커피", "coffee", "café"},
}

# (옵션) 내부 시설 샘플 데이터 (있으면 안내 멘트 강화)
PLACE_KB = {
    # 키는 표준명
    "화장실": {
        "ko": "화장실은 1층 중앙 로비 오른쪽 20m, 2층 전시실 입구 왼쪽에 있어요.",
        "en": "Restrooms: 1F central lobby (right ~20m) and 2F near the gallery entrance (left).",
        "ja": "トイレは1階ロビー中央の右側に20m、2階展示室入口の左側です。",
    }
}

def normalize_text(text: str) -> str:
    t = text.strip()
    for wrong, right in NORMALIZE_MAP.items():
        t = t.replace(wrong, right)
    return t

def canonical_place(word: str) -> Optional[str]:
    w = word.lower()
    for k, variants in PLACE_ALIASES.items():
        if w in {v.lower() for v in variants} or k.lower() in w:
            return k
    return None

def recommend_postprocess(result: IntentResult, lang: str, context: List[Turn]) -> IntentResult:
    """LLM 결과에 규칙 기반 보강(슬롯 보정, 답변 추천, 액션 생성)."""
    r = result

    # slots 보정
    if r.slots is None:
        r.slots = {}

    # 최근 문맥에서 빠진 값 보충(예: 직전 목적지/장소)
    def find_last_slot(key: str) -> Optional[Any]:
        for t in reversed(context):
            if t.result and t.result.slots and key in t.result.slots:
                return t.result.slots[key]
        return None

    # PLACE_INFO: place 정규화/보강
    if r.intent == Intent.PLACE_INFO:
        place = r.slots.get("place")
        if isinstance(place, str):
            place_std = canonical_place(place) or place
        else:
            place_std = find_last_slot("place")

        if place_std:
            r.slots["place"] = place_std
            info = PLACE_KB.get(place_std)
            if info:
                # 명확히 아는 장소면 지식 베이스 멘트 추천
                r.speak = info.get(lang, info.get("ko", r.speak))
                r.confidence = max(r.confidence, 0.8)
                r.actions.append({"service": "INFO", "name": "place_info", "params": {"place": place_std}})
            else:
                # 모르는 장소면 위치 추가 질문
                r.ask_user = r.ask_user or ( "어느 층/구역을 찾으세요?" if lang=="ko" else "Which floor/area?" )
                r.actions.append({"service": "INFO", "name": "ask_place_detail", "params": {"place": place_std}})
        else:
            # 장소가 비어있으면 적극 질문
            r.ask_user = r.ask_user or ( "어느 시설을 말씀하시나요? 예: 화장실, 안내데스크" if lang=="ko" else "Which facility? e.g., restroom, info desk" )
            r.confidence = min(r.confidence, 0.4)

    # ROUTE_REQUEST: 출발/도착 누락 시 문맥/추가질문
    if r.intent == Intent.ROUTE_REQUEST:
        src = r.slots.get("from") or find_last_slot("from")
        dst = r.slots.get("to") or r.slots.get("destination") or find_last_slot("to")
        if src: r.slots["from"] = src
        if dst: r.slots["to"] = dst

        if not dst:
            r.ask_user = r.ask_user or ( "목적지를 알려주세요." if lang=="ko" else "Where would you like to go?" )
            r.confidence = min(r.confidence, 0.4)
        else:
            r.actions.append({"service":"NAV","name":"plan_route","params":{"from":src,"to":dst}})
            if not r.speak or r.speak=="무엇을 도와드릴까요?":
                r.speak = (f"{dst}까지 경로를 안내할게요." if lang=="ko" else f"I'll guide you to {dst}.")

    # TRANSLATE: toLang 기본값/질문
    if r.intent == Intent.TRANSLATE:
        if not r.slots.get("toLang"):
            r.ask_user = r.ask_user or ( "어떤 언어로 번역할까요?" if lang=="ko" else "Which target language?" )

    # SET_LANGUAGE: 언어 코드 표준화
    if r.intent == Intent.SET_LANGUAGE and "lang" in r.slots:
        code = str(r.slots["lang"]).lower()
        aliases = {"kr":"ko","korean":"ko","en":"en","english":"en","jp":"ja","japanese":"ja","ja":"ja","zh":"zh"}
        r.slots["lang"] = aliases.get(code, code)

    # 알람/비상: 응급 멘트 보강
    if r.intent == Intent.ALARM_EMERGENCY:
        if lang=="ko":
            r.speak = "비상 호출을 접수했습니다. 잠시만 기다려 주세요."
        else:
            r.speak = "Emergency request received. Please wait a moment."
        r.actions.append({"service":"CALL","name":"notify_security","params":{}})

    # 기본: speak 없으면 기본 멘트
    if not r.speak:
        r.speak = "무엇을 도와드릴까요?" if lang=="ko" else "How can I help you?"

    # confidence 클램프
    r.confidence = max(0.0, min(1.0, float(r.confidence or 0.5)))
    return r

# --------------------------
# LLM 프롬프트(엄격 JSON)
# --------------------------
SYSTEM_PROMPT = """당신은 '관광 안내 로봇'의 플래너입니다.
아래 JSON 스키마에 '정확히 맞는' JSON만 출력하세요. 코드블록/설명/접두어 금지.
필수 키: intent, slots, actions, speak, ask_user, confidence
intent 값(택1): [ROUTE_REQUEST, PLACE_INFO, CALL_STAFF, TRANSLATE, SET_LANGUAGE, SMALL_TALK, ALARM_EMERGENCY, UNKNOWN]
slots: 의도 실행에 필요한 값 (예: {"to":"미술관", "from":"로비"}, {"place":"화장실"}, {"toLang":"en"}, {"lang":"ko"})
actions: 실행 단계 목록 (예: [{"service":"NAV","name":"plan_route","params":{"to":"...","from":"..."}}])
speak: 사용자에게 즉시 말할 한 문장 (요청 언어)
ask_user: 정보 부족 시 추가로 물을 한 문장 (없으면 빈 문자열)
confidence: 0~1 사이 숫자
JSON만 출력하세요.
"""

FEW_SHOTS = [
    # 질의, 언어, 기대 JSON(예시)
    ("화장실 어디에 있어?", "ko", {
        "intent":"PLACE_INFO",
        "slots":{"place":"화장실"},
        "actions":[{"service":"INFO","name":"place_info","params":{"place":"화장실"}}],
        "speak":"화장실 위치를 안내해 드릴게요.",
        "ask_user":"",
        "confidence":0.75
    }),
    ("Please translate: '안녕하세요' to English", "en", {
        "intent":"TRANSLATE",
        "slots":{"text":"안녕하세요","toLang":"en"},
        "actions":[{"service":"NLP","name":"translate","params":{"toLang":"en"}}],
        "speak":"I'll translate it to English.",
        "ask_user":"",
        "confidence":0.8
    }),
    ("길 좀 알려줘. 전시실 가고 싶어.", "ko", {
        "intent":"ROUTE_REQUEST",
        "slots":{"to":"전시실"},
        "actions":[{"service":"NAV","name":"plan_route","params":{"to":"전시실"}}],
        "speak":"전시실까지 경로를 안내할게요.",
        "ask_user":"",
        "confidence":0.8
    }),
]

FEW_SHOTS.append({
    "user": "지도 보여줘",
    "lang": "ko",
    "out": {
        "intent":"PLACE_INFO",
        "slots":{"target":"map","map_key":"default"},
        "actions":[{"service":"INFO","name":"show_image","params":{"map_key":"default"}}],
        "speak":"지도를 화면에 표시할게요.",
        "ask_user":"",
        "confidence":0.85
    }
})

# llm_server.py
MAP_FILES = {
    "default": "venue_map.png",
    "1f": "floor1.png",
    "2f": "floor2.png",
}
def build_map_url(map_key: str | None) -> str:
    k = (map_key or "default").lower().replace(" ", "")
    fname = MAP_FILES.get(k, MAP_FILES["default"])
    return f"/static/maps/{fname}"

def recommend_postprocess(result: IntentResult, lang: str, context: List[Turn], user_text: str = "") -> IntentResult:
    r = result
    # ... (기존 보강 로직)

    # 사용자가 '지도'를 직접 언급했는데 액션이 비어 있으면 생성
    if ("지도" in user_text or "map" in user_text.lower()):
        has_show = any(a.get("service")=="INFO" and a.get("name")=="show_image" for a in r.actions)
        if not has_show:
            mk = r.slots.get("map_key") if isinstance(r.slots, dict) else None
            r.actions.append({"service":"INFO","name":"show_image","params":{"map_key": mk or "default"}})
            if not r.speak or r.speak == "무엇을 도와드릴까요?":
                r.speak = "지도를 화면에 표시할게요."

    # show_image에 url 채우기
    for a in r.actions:
        if a.get("service")=="INFO" and a.get("name")=="show_image":
            params = a.setdefault("params",{})
            if "url" not in params:
                params["url"] = build_map_url(params.get("map_key") or r.slots.get("map_key"))

    return r

def build_messages(user: str, lang: str, context: List[Turn]) -> List[Dict[str, str]]:
    # 최근 문맥 요약(슬롯 위주)
    ctx_lines = []
    for t in context[-4:]:
        try:
            i = t.result.intent.value if isinstance(t.result.intent, Enum) else str(t.result.intent)
            ctx_lines.append(f"- intent={i}, slots={json.dumps(t.result.slots, ensure_ascii=False)}")
        except Exception:
            pass
    ctx_text = "\n".join(ctx_lines) if ctx_lines else "(none)"

    msgs: List[Dict[str,str]] = [
        {"role":"system","content": SYSTEM_PROMPT + f"\n[State] ui_lang={lang}\n[RecentTurns]\n{ctx_text}\n[Output] JSON only."},
    ]
    # few-shot
    for q, qlang, outj in FEW_SHOTS:
        msgs.append({"role":"user","content": f"[lang={qlang}] {q}"})
        msgs.append({"role":"assistant","content": json.dumps(outj, ensure_ascii=False)})
    # 현재 사용자 입력
    msgs.append({"role":"user","content": f"[lang={lang}] {user}"})
    return msgs

# --------------------------
# 모델 로딩
# --------------------------
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
        # 일반 모델 대비(비권장이지만 폴백)
        full = ""
        for m in messages:
            role = m["role"]
            full += f"<|{role}|>\n{m['content'].strip()}\n"
        full += "<|assistant|>\n"
        return full

def call_llm(prompt: str) -> str:
    # Qwen은 eos와 <|im_end|> 모두 사용할 수 있음
    terminators: List[int] = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != tokenizer.eos_token_id:
            terminators.append(im_end)
    except Exception:
        pass

    out = gen(
        prompt,
        max_new_tokens=220,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=terminators if terminators else None,
    )
    text = out[0]["generated_text"]
    # 프롬프트 프리픽스 제거
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# --------------------------
# 안전 JSON 파싱 유틸
# --------------------------
# JSON_REGEX = re.compile(r"\{(?:[^{}]|(?R))*\}", re.S)

def extract_json_block(s: str) -> str:
    """
    문자열 s에서 첫 번째 JSON 객체({ ... }) 블록을 균형 잡힌 중괄호 기준으로 추출.
    문자열 내부의 '{' '}'는 따옴표/이스케이프를 고려해 무시합니다.
    못 찾으면 s 그대로(또는 첫 '{'부터 끝까지) 반환.
    """
    if not s:
        return s
    # 코드펜스/앞뒤 노이즈 제거
    s = s.replace("```json", "").replace("```", "").strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    # 닫는 괄호 못 찾은 경우: 시작부터 끝까지 반환(파서가 실패하면 2차 복구 프롬프트가 처리)
    return s[start:]

def coerce_result(obj: Dict[str, Any]) -> IntentResult:
    # 누락 필드 보정
    obj.setdefault("intent", "UNKNOWN")
    obj.setdefault("slots", {})
    obj.setdefault("actions", [])
    obj.setdefault("speak", "무엇을 도와드릴까요?")
    obj.setdefault("ask_user", "")
    obj.setdefault("confidence", 0.5)
    # 타입 보정
    if obj["slots"] is None: obj["slots"] = {}
    if obj["actions"] is None: obj["actions"] = []
    try:
        return IntentResult(**obj)
    except ValidationError as e:
        log.warning(f"[파싱/검증 보정 실패] {e}")
        return IntentResult()

def run_intent_llm(user: str, lang: str, context: List[Turn]) -> IntentResult:
    """LLM 1차 호출 → JSON 추출 → 검증 실패 시 복구 프롬프트로 2차 시도."""
    messages = build_messages(user, lang, context)
    prompt = apply_chat_template(messages)
    raw = call_llm(prompt)
    log.info(f"[LLM raw] {raw[:500]}{'...' if len(raw)>500 else ''}")

    # 1차 파싱
    try:
        block = extract_json_block(raw)
        data = json.loads(block)
        res = coerce_result(data)
        res._raw = raw
        return res
    except Exception as e1:
        log.warning(f"[1차 파싱 실패] {e1}")

    # 2차: JSON만 남기도록 복구 프롬프트
    repair_msgs = [
        {"role":"system","content": "Fix to valid JSON. Output JSON ONLY. No code fences."},
        {"role":"user","content": f"Text:\n{raw}\n---\nReturn valid JSON matching keys: intent, slots, actions, speak, ask_user, confidence"}
    ]
    repair_prompt = apply_chat_template(repair_msgs)
    fixed = call_llm(repair_prompt)
    try:
        block = extract_json_block(fixed)
        data = json.loads(block)
        res = coerce_result(data)
        res._raw = raw
        return res
    except Exception as e2:
        log.error(f"[2차 파싱 실패] {e2}")
        # 완전 폴백
        return IntentResult(_raw=raw)

# --------------------------
# FastAPI
# --------------------------
app = FastAPI(title="TourBot LLM Server", version="1.0")

@app.get("/health")
def health():
    try:
        _ = gen("ping", max_new_tokens=1, do_sample=False)
        return {"status":"ok"}
    except Exception as e:
        return JSONResponse({"status":"error","detail":str(e)}, status_code=503)

@app.post("/intent")
def infer_intent(req: IntentReq):
    user0 = req.text or ""
    lang = req.lang or "ko"
    sid = req.session_id or "default"

    # 0) STT 오인식/노이즈 간단 정규화
    user = normalize_text(user0)

    # 1) 문맥 로드
    ctx = MEM.get_context(sid)

    # 2) LLM 호출 → JSON 스키마 결과
    result = run_intent_llm(user, lang, ctx)

    # 3) 추천 후처리(룰/지식) + 문맥 반영
    # result = recommend_postprocess(result, lang, ctx)  # 기존
    result = recommend_postprocess(result, lang, ctx, user_text=user)  # ← 이렇게
    MEM.put(sid, user, result)

    return result.model_dump()

@app.post("/reset")
def reset_session(req: Dict[str, str]):
    sid = (req or {}).get("session_id") or "default"
    MEM.reset(sid)
    return {"status":"ok","session_id":sid}

# --------------------------
# 로컬 테스트용 메인
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_server:app", host="0.0.0.0", port=9100, reload=False)


"""
실행 예:
cd /data/bootcamp/bootcamp/work_ID1/apps.250728_copy/servers/llm
conda activate llm-server
CUDA_VISIBLE_DEVICES=1 uvicorn llm_server:app --host 0.0.0.0 --port 9100
"""
