from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, io, tempfile, re
import numpy as np
import soundfile as sf
import librosa
import torch
import whisper
import webrtcvad

app = FastAPI()

# ── Whisper 모델 준비
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = os.environ.get("WHISPER_MODEL","medium")
wh_model = whisper.load_model(MODEL, device=DEVICE)

# ── VAD 파라미터
VAD_SR      = 16000
FRAME_MS    = 20
AGGR        = int(os.environ.get("VAD_AGGR", 2))   # 0~3
START_HITS  = 5     # 최근 8프레임 중 N 프레임 이상이면 시작
HISTORY     = 8
END_HANG_MS = 1000  # 말끝 행오버
PREROLL_MS  = 150
POSTROLL_MS = 300
MAX_SEG_MS  = 8000  # 세그먼트 최대 길이

def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = np.mean(x, axis=1 if x.shape[0] < x.shape[1] else 0)
    if sr != VAD_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=VAD_SR)
    # float32 -1~1
    x = x.astype(np.float32, copy=False)
    return x

def frame_iter(pcm16: np.ndarray, frame_len: int):
    # yield bytes of length frame_len*2
    for i in range(0, len(pcm16), frame_len):
        chunk = pcm16[i:i+frame_len]
        if len(chunk) < frame_len:
            break
        yield chunk.tobytes()

def vad_segments(pcm16: np.ndarray):
    vad = webrtcvad.Vad(AGGR)
    frm = int(VAD_SR * FRAME_MS / 1000)
    preroll = int(VAD_SR * PREROLL_MS / 1000)
    postroll = int(VAD_SR * POSTROLL_MS / 1000)
    end_hang = int(VAD_SR * END_HANG_MS / 1000)
    max_seg  = int(VAD_SR * MAX_SEG_MS / 1000)

    speech_started = False
    start_idx = None
    last_speech_idx = None
    hits = []
    i = 0

    for fb in frame_iter(pcm16, frm):
        is_speech = vad.is_speech(fb, VAD_SR)
        hits.append(1 if is_speech else 0)
        if len(hits) > HISTORY:
            hits.pop(0)

        if not speech_started:
            if sum(hits) >= START_HITS:
                speech_started = True
                start_idx = max(0, i*frm - preroll)
                last_speech_idx = i*frm
        else:
            if is_speech:
                last_speech_idx = i*frm
            reached_hang = (i*frm - last_speech_idx) >= end_hang
            too_long     = (i*frm - start_idx)      >= max_seg
            if reached_hang or too_long:
                end_idx = min(len(pcm16), last_speech_idx + postroll)
                yield (start_idx, end_idx)
                speech_started = False
                start_idx = None
                last_speech_idx = None
                hits.clear()
        i += 1

    if speech_started and last_speech_idx is not None:
        end_idx = min(len(pcm16), last_speech_idx + postroll)
        yield (start_idx, end_idx)

def clean_stt_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^[.,\s]+|[.,\s]+$", "", text)
    return text

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        # 1) 파일 저장 없이 메모리에서 읽기
        data = await file.read()
        with sf.SoundFile(io.BytesIO(data)) as f:
            x = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate

        # 2) 16k mono 변환 → int16 for VAD
        x16k = to_16k_mono(x, sr)
        pcm16 = np.clip(x16k * 32768.0, -32768, 32767).astype(np.int16)

        # 3) VAD 세그먼트 나누기
        texts = []
        seg_count = 0
        for s, e in vad_segments(pcm16):
            seg_count += 1
            seg = x16k[s:e]  # float32 -1~1
            if len(seg) < int(0.3 * VAD_SR):
                continue  # 300ms 미만 버림

            # 4) Whisper 추론(세그먼트 단위)
            #   - condition_on_previous_text=False: 조각간 영향 최소화
            res = wh_model.transcribe(seg, language="ko", condition_on_previous_text=False, temperature=0)
            # 5) 무음/헛소리 필터(첫 segment 기준 no_speech_prob)
            if res.get("segments"):
                first = res["segments"][0]
                if first.get("no_speech_prob", 0) > 0.8 and len(res.get("text","" ).strip()) < 2:
                    continue
            text = clean_stt_text(res.get("text",""))
            if text:
                texts.append(text)

        joined = " ".join(texts).strip()
        return JSONResponse({"text": joined})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "device": DEVICE}