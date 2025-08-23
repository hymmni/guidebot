conda activate whisper-server
CUDA_VISIBLE_DEVICES=2 uvicorn whisper_server:app --host 0.0.0.0 --port 9000 --reload --log-level debug
