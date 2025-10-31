# llm_server.py — Qwen 기반 인텐트 서버 (+규칙 폴백)
# 실행 예:
#   CUDA_VISIBLE_DEVICES=0 uvicorn llm_server:app --host 0.0.0.0 --port 9100
import os, re, json, time, logging, threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# Qwen 로딩 (없으면 예외 → 아래 규칙 폴백 사용)
try:
    from transformers import AutoTokenizer, pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("llm_server")

# ===============================
# 설정
# ===============================
MODEL_NAME  = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")  # VRAM 여유 시 14B 가능
DEVICE_MAP  = os.environ.get("LLM_DEVICE_MAP", "auto")
MAX_TOKENS  = int(os.environ.get("MAX_NEW_TOKENS", "220"))

# ===============================
# FastAPI
# ===============================
app = FastAPI(title="Qwen Intent Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ===============================
# 데이터/스키마
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
# 유틸: 언어 감지/정규화
# ===============================
def detect_lang(text: str) -> str:
    t = text.strip()
    if re.search(r"[\u3040-\u30ff]", t): return "ja"  # 히라가나/가타카나
    if re.search(r"[\uac00-\ud7af]", t): return "ko"  # 한글
    if len(re.findall(r"[\u4e00-\u9fff]", t)) >= 3: return "zh"  # 한자 많음 → 중문
    return "en"

def norm_floor(text: str) -> Optional[str]:
    if not text: return None
    t = text.lower().replace(" ", "")
    m = re.search(r"지하\s*([0-9]+)\s*층", t)
    if m: return f"b{m.group(1)}"
    m = re.search(r"([0-9]+)\s*층", t)
    if m: return f"{m.group(1)}f"
    m = re.search(r"\b(b[0-9]+|[0-9]+f)\b", t)
    if m: return m.group(1).lower()
    return None

def speak_in(lang: str, ko: str, en: str, ja: str = "", zh: str = "") -> str:
    if lang == "ko": return ko
    if lang == "ja": return ja or en
    if lang == "zh": return zh or en
    return en

# 금관 콘텐츠(LLM이 내려줌)
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
# 시스템 프롬프트/샘플
# ===============================
SYSTEM_PROMPT = """You are an intent parser for a kiosk in Korean named "에스뽀".
Return ONLY strict JSON with keys: intent, slots, actions, speak, ask_user, confidence.
- intent ∈ [SHOW_MAP, ROUTE_REQUEST, PLACE_INFO, CALL_STAFF, TRANSLATE, SET_LANGUAGE, SMALL_TALK, ALARM_EMERGENCY, UNKNOWN]
- slots may include: floor ('1f','2f','b1'...), destination/to, from, lang ('ko','en','ja','zh'...)
- actions: array of {service, name, params}. For SHOW_MAP include params.map_key if possible.
- speak: short, kind, and cheerful reply (<=20 words) in the SAME LANGUAGE as the user's utterance.
- ask_user: if info is missing, one cute and friendly clarifying question in the user's language; else empty.
- confidence: 0..1 number.

Rules:
- Your name is "에스뽀". If user uses another language, answer in that language.
- If asked "show map" without a floor, DO NOT guess. Ask which floor (B1~10F) and propose a floor choice UI.
- Your personality is like a very friendly, cute, and helpful friend! Please end your sentences with cute endings like `~용`, `~죠`, `~뿅!`, or `~이요`. Use emojis often!
- Output JSON only, no code fences.
- 상대방이 영어로 질문하면 무조건 영어로 대답할 것
- 말 끝에 용 붙이지 말것
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
# 모델 로딩 (가능할 때만)
# ===============================
if HF_AVAILABLE:
    try:
        log.info(f"Loading model: {MODEL_NAME} (device_map={DEVICE_MAP})")
        from transformers import AutoTokenizer, pipeline
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
        MODEL_READY = True
    except Exception as e:
        log.warning(f"HF model load failed: {e}")
        MODEL_READY = False
else:
    MODEL_READY = False

def apply_chat_template(messages: List[Dict[str,str]]) -> str:
    if not MODEL_READY:
        # 사용하지 않음
        return ""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        txt=""
        for m in messages:
            txt += f"<|{m['role']}|>\n{m['content'].strip()}\n"
        txt += "<|assistant|>\n"
        return txt

def call_llm(prompt: str) -> str:
    if not MODEL_READY:
        return ""
    terminators=[]
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != getattr(tokenizer, "eos_token_id", None):
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

def rule_fallback(text: str, lang: str) -> Dict[str, Any]:
    t = text.lower()
    # 비상
    if any(k in t for k in ["비상","도움","살려","help","emergency","警报"]):
        return {
            "intent":"ALARM_EMERGENCY","slots":{},
            "actions":[{"service":"CALL","name":"notify_security","params":{}}],
            "speak": speak_in(lang, "도움이 필요하신가요? 직원을 호출합니다.", "Calling staff for help.", "スタッフを呼びます。"),
            "ask_user":"", "confidence":0.9
        }
    # 직원 호출
    if any(k in t for k in ["직원","호출","call staff","staff"]):
        return {
            "intent":"CALL_STAFF","slots":{},
            "actions":[{"service":"CALL","name":"notify_staff","params":{}}],
            "speak": speak_in(lang,"직원을 호출할게요.","I'll call a staff member.","スタッフを呼びます。"),
            "ask_user":"", "confidence":0.95
        }
    # 언어 변경
    if any(k in t for k in ["한국어","영어","english","일본어","japanese","日本語","중국어","中文","chinese"]):
        target = "ko"
        if any(k in t for k in ["영어","english"]): target = "en"
        elif any(k in t for k in ["일본어","japanese","日本語"]): target = "ja"
        elif any(k in t for k in ["중국어","中文","chinese"]): target = "zh"
        return {
            "intent":"SET_LANGUAGE","slots":{"lang":target},
            "actions":[{"service":"UI","name":"set_lang","params":{"lang": target}}],
            "speak": speak_in(target,"화면 언어를 한국어로 바꿨어요.","Language set to English.","言語を日本語に変更しました。"),
            "ask_user":"", "confidence":0.98
        }
    # 지도
    if any(k in t for k in ["지도","맵","map"]):
        fk = norm_floor(t)
        if not fk:
            prompt = speak_in(lang,
                "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?",
                "We have B1 to 10F. Which floor map would you like?",
                "地下1階から地上10階まで。どの階の地図を表示しますか？")
            return {
                "intent":"SHOW_MAP","slots":{},
                "actions":[{"service":"UI","name":"choose_floor",
                            "params":{"floors":["b1"]+[f"{i}f" for i in range(1,11)],"prompt":prompt}}],
                "speak":prompt,"ask_user":"", "confidence":0.85
            }
        return {
            "intent":"SHOW_MAP","slots":{"floor":fk},
            "actions":[{"service":"INFO","name":"show_map","params":{"map_key": fk}}],
            "speak": speak_in(lang, "지도를 보여드릴게요.","Showing the map.","地図を表示します。"),
            "ask_user":"", "confidence":0.95
        }
    # 길 안내
    if any(k in t for k in ["길 안내","가는 길","route","네비","navigate"]):
        return {
            "intent":"ROUTE_REQUEST","slots":{"from":"로비","to":"전시실"},
            "actions":[{"service":"NAV","name":"plan_route","params":{"from":"로비","to":"전시실"}}],
            "speak": speak_in(lang,"길 안내를 시작할게요.","Starting route guidance.","経路案内を開始します。"),
            "ask_user":"", "confidence":0.9
        }
    # 정보 → 선택 패널
    if any(k in t for k in ["정보","설명","안내","explain","info"]):
        prompt = speak_in(lang, "무엇을 설명해드릴까요?","What would you like to know about?","何について説明しますか？")
        return {
            "intent":"PLACE_INFO","slots":{},
            "actions":[{
                "service":"UI","name":"choice",
                "params":{
                    "title": speak_in(lang,"정보","Information","情報"),
                    "prompt": prompt,
                    "options":[
                        {"label": speak_in(lang,"금관","Geumgwan (Gold Crown)","金冠"),
                         "say": speak_in(lang,"금관 설명해줘","Tell me about the Geumgwan.","金冠について説明して")}
                    ]
                }
            }],
            "speak": prompt, "ask_user":"", "confidence":0.88
        }
    # 금관
    if any(k in t for k in ["금관","geumgwan","gold crown","金冠"]):
        return {
            "intent":"PLACE_INFO","slots":{"topic":"geumgwan"},
            "actions":[
                {"service":"INFO","name":"show_artifact","params":{
                    "title": GEUMGWAN["title"], "url": GEUMGWAN["url"], "desc": GEUMGWAN["desc"]
                }},
                {"service":"TTS","name":"speak_detail","params":{"text": GEUMGWAN["desc"]}}
            ],
            "speak": speak_in(lang,"금관에 대해 알려드릴게요.","Let me tell you about the Geumgwan.","金冠についてご案内します。"),
            "ask_user":"", "confidence":0.96
        }
    # 스몰톡(이름)
    if any(k in t for k in ["에스뽀","에스포","espo","spo"]):
        return {
            "intent":"SMALL_TALK","slots":{},
            "actions":[],
            "speak": speak_in(lang,"저는 에스뽀예요. 무엇을 도와드릴까요?","I'm Espo. How can I help?","エスポです。ご用件は？"),
            "ask_user":"", "confidence":0.8
        }
    return {
        "intent":"UNKNOWN","slots":{},"actions":[],
        "speak": speak_in(lang,"요청을 이해하지 못했어요.","Sorry, I didn't catch that.","すみません、よく分かりませんでした。"),
        "ask_user":"", "confidence":0.4
    }
    # 작별/종료/홈 이동
    if any(k in t for k in ["안녕", "수고", "고마워", "감사", "bye", "바이", "그만", "끝", "종료", "취소", "홈", "처음으로", "초기 화면", "home"]):
        return {
            "intent": "SMALL_TALK",
            "slots": {"lang": lang},
            "actions": [
                {"service": "UI", "name": "reset_ui", "params": {"goodbye": True}}
            ],
            "speak": (
                "이용해 주셔서 감사합니다. 초기 화면으로 돌아갈게요."
                if lang == "ko" else
                ("Thanks for visiting. Returning to the home screen." if lang == "en"
                 else "ご利用ありがとうございました。ホーム画面に戻ります。")
            ),
            "ask_user": "",
            "confidence": 0.98
        }

# ===============================
# 보강/후처리
# ===============================
def normalize_floor_slot(res: Dict[str, Any], user_text: str):
    floor = (res.get("slots") or {}).get("floor")
    nf = norm_floor(floor or user_text)
    if nf:
        res.setdefault("slots", {})["floor"] = nf
        # SHOW_MAP 액션에 map_key 보강
        added = False
        for a in res.get("actions", []):
            if a.get("service")=="INFO" and a.get("name") in ("show_map","map","show_image"):
                a.setdefault("params", {}).setdefault("map_key", nf)
                added = True
        if res.get("intent")=="SHOW_MAP" and not added:
            res.setdefault("actions", []).append({"service":"INFO","name":"show_map","params":{"map_key": nf}})
        if not res.get("speak"):
            if nf.endswith("f") and nf[:-1].isdigit():
                res["speak"] = f"{int(nf[:-1])}층 지도를 보여드릴게요."
            elif nf.startswith("b") and nf[1:].isdigit():
                res["speak"] = f"지하 {int(nf[1:])}층 지도를 보여드릴게요."
            else:
                res["speak"] = "지도를 보여드릴게요."
    else:
        if res.get("intent")=="SHOW_MAP":
            prompt = "지하 1층부터 지상 10층까지 있어요. 몇 층 지도를 보여드릴까요?"
            res["speak"] = res.get("speak") or prompt
            res.setdefault("actions", []).append({
                "service":"UI","name":"choose_floor",
                "params":{"floors":["b1"]+[f"{i}f" for i in range(1,11)], "prompt": prompt}
            })

# ===============================
# 엔드포인트
# ===============================
@app.get("/health")
def health():
    return {"ok": True, "model_ready": bool(MODEL_READY)}

@app.post("/intent")
async def intent(req: Request):
    data = await req.json()
    text = (data.get("text") or "").strip()
    sid  = data.get("session_id","default")
    lang = (data.get("lang") or detect_lang(text) or "ko")

    # 1) LLM 호출(가능 시)
    result_obj: Optional[IntentResult] = None
    if MODEL_READY and text:
        try:
            messages = build_messages(text)
            prompt   = apply_chat_template(messages)
            raw      = call_llm(prompt)
            block    = extract_json_block(raw)
            obj      = json.loads(block)
            result_obj = IntentResult(**obj)
        except Exception as e:
            log.warning(f"LLM parse failed, use fallback: {e}")

    # 2) 폴백/후처리
    if result_obj is None:
        res = rule_fallback(text, lang)
    else:
        res = result_obj.model_dump()

    # 언어/층 보강
    res.setdefault("slots", {})
    res["slots"].setdefault("lang", lang)
    normalize_floor_slot(res, text)

    # 정보 요청인데 choice가 없다면 기본 옵션(금관)을 붙여서 유도
    t = text.lower()
    if any(k in t for k in ["정보","설명","안내","explain","info"]):
        if not any(a.get("service")=="UI" and a.get("name")=="choice" for a in res.get("actions", [])):
            res.setdefault("actions", []).append({
                "service":"UI","name":"choice",
                "params":{
                    "title":  speak_in(lang,"유물 선택","Artifact Selection","遺物を選択"),
                    "prompt": speak_in(lang,"어떤 유물을 설명해드릴까요?",
                                            "Which artifact would you like to know about?",
                                            "どの遺物について説明しますか？"),
                    "options":[
                        {
                        "label": speak_in(lang,"신라 금관","Silla Gold Crown","新羅金冠"),
                        "say":   speak_in(lang,"신라 금관 설명해줘",
                                                    "Tell me about the Silla Gold Crown.",
                                                    "新羅金冠について説明して")
                        },
                        {
                        "label": speak_in(lang,"금동미륵보살반가사유상",
                                                    "Gilt-bronze Maitreya in Meditation",
                                                    "金銅彌勒菩薩半跏思惟像"),
                        "say":   speak_in(lang,"금동미륵보살반가사유상 설명해줘",
                                                    "Tell me about the Gilt-bronze Maitreya.",
                                                    "金銅彌勒菩薩半跏思惟像について説明して")
                        },
                        {
                        "label": speak_in(lang,"청자 상감운학문 매병",
                                                    "Celadon Inlaid Maebyeong (Cloud & Crane)",
                                                    "青磁象嵌雲鶴文梅瓶"),
                        "say":   speak_in(lang,"청자 상감운학문 매병 설명해줘",
                                                    "Tell me about the celadon inlaid maebyeong.",
                                                    "青磁象嵌雲鶴文梅瓶について説明して")
                        },
                        {
                        "label": speak_in(lang,"경천사 십층석탑",
                                                    "Ten-story Stone Pagoda of Gyeongcheonsa",
                                                    "敬天寺十層石塔"),
                        "say":   speak_in(lang,"경천사 십층석탑 설명해줘",
                                                    "Tell me about the Gyeongcheonsa pagoda.",
                                                    "敬天寺十層石塔について説明して")
                        }
                    ]
                }
            })
        res["speak"] = res.get("speak") or speak_in(
            lang,"어떤 유물을 설명해드릴까요?",
                "Which artifact would you like to know about?",
                "どの遺物について説明しますか？"
        )
    # 금관 키워드면 확실히 이미지+긴 설명 TTS 부여
    if any(k in t for k in ["금관","geumgwan","gold crown","金冠"]):
        if not any(a.get("name")=="show_artifact" for a in res.get("actions", [])):
            res.setdefault("actions", []).append({
                "service":"INFO","name":"show_artifact",
                "params":{"title":GEUMGWAN["title"],"url":GEUMGWAN["url"],"desc":GEUMGWAN["desc"]}
            })
        if not any(a.get("name")=="speak_detail" for a in res.get("actions", [])):
            res.setdefault("actions", []).append({
                "service":"TTS","name":"speak_detail","params":{"text": GEUMGWAN["desc"]}}
            )
        res["speak"] = res.get("speak") or speak_in(lang,"금관에 대해 알려드릴게요.","Let me tell you about the Geumgwan.","金冠についてご案内します。")

    # 3) 저장/반환
    try:
        MEM.put(sid, text, IntentResult(**res))
    except Exception:
        pass
    return JSONResponse(res)

@app.post("/reset")
async def reset_session(req: Request):
    data = await req.json()
    sid = data.get("session_id","default")
    MEM.reset(sid)
    return {"status":"ok","session_id":sid}
