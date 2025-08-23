# servers/llm/llm_server.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline
import torch, logging, json, re

import sys, os
# 현재 파일 기준으로 2단계 상위 폴더(= apps.250728_copy) 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.intent import IntentResult

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentReq(BaseModel):
    text: str
    lang: str = "ko"

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"   # or "Qwen/Qwen2.5-7B-Instruct"

# --- 모델 로딩 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
pipe = pipeline(
    task="text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"  # 여러 GPU면 자동 분산
)
pipe.model.eval()

SYSTEM_PROMPT = """당신은 관광 안내 로봇의 플래너입니다.
오직 아래 JSON 스키마(IntentResult)에 맞는 JSON만 출력하세요. 설명/코드블럭/문장 금지.
필드:
- intent: [ROUTE_REQUEST, PLACE_INFO, CALL_STAFF, TRANSLATE, SET_LANGUAGE, SMALL_TALK, ALARM_EMERGENCY, UNKNOWN]
- slots: 의도 실행에 필요한 값 (예: destination, place, toLang 등)
- actions: 실행 단계 목록 (예: {"service":"NAV","name":"plan_route","params":{"to":"..."}})
- speak: 사용자에게 말할 1문장 (lang에 맞춰)
- ask_user: 정보 부족 시 물어볼 1문장
- confidence: 0~1
JSON만 출력하세요.
"""

def build_prompt(user: str, lang: str = "ko") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n[State] lang={lang}\n[Output] JSON only."},
        {"role": "user", "content": user}
    ]
    # Qwen은 chat 템플릿 지원
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def call_llm(prompt: str) -> str:
    # Qwen은 eos(</s>)와 <|im_end|> 둘 다 사용할 수 있음
    terminators = [tokenizer.eos_token_id]
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != tokenizer.eos_token_id:
            terminators.append(im_end)
    except Exception:
        pass

    out = pipe(
        prompt,
        max_new_tokens=220,
        do_sample=False,    # 파싱 안정성 우선
        temperature=0.0,
        top_p=1.0,
        eos_token_id=terminators
    )
    gen = out[0]["generated_text"]
    if gen.startswith(prompt):
        gen = gen[len(prompt):]
    return gen.strip()

def extract_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    raw = m.group(0) if m else s
    return json.loads(raw)

@app.get("/health")
def health():
    try:
        _ = pipe("테스트", max_new_tokens=1, do_sample=False)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)

@app.post("/intent")
def infer_intent(req: IntentReq):
    user = (req.text or "").strip()
    prompt = build_prompt(user, req.lang)
    raw = call_llm(prompt)
    logger.info(f"[LLM raw] {raw[:300]}{'...' if len(raw)>300 else ''}")

    try:
        data = extract_json(raw)
        result = IntentResult(**data)  # 스키마 검증
    except Exception as e:
        logger.warning(f"[파싱 실패] {e}")
        result = IntentResult(speak="무엇을 도와드릴까요?")

    return result.model_dump()


'''
cd /data/bootcamp/bootcamp/work_ID1/apps.250728_copy/servers/llm
conda activate llm-server
CUDA_VISIBLE_DEVICES=1 uvicorn llm_server:app --port 9100
'''
