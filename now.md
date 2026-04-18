# Now — 现在在做什么

## 当前任务
- 正确性审查 + 端到端 mock 评测完成。等待在真实 4x5090 + Qwen3-32B 上做最终验证。

## 本地环境速查
- conda env: `inference`（Python 3.11）
- Redis: Memurai Developer（Windows 服务）
- inference python: `C:\Users\Keyi\miniconda3\envs\inference\python.exe`
- 启动 ubiservice：`python start_ubiservice_local.py`（用 inference python）
- 启动 mock vLLM：`python mock_vllm.py --port 8000 [--latency-ms N] [--deterministic]`
- 跑 logprob 正确性测试：`python tests/test_logprob_correctness.py`（需要 --deterministic 的 mock 在 8000）
- 端到端评测：`python run_eval_windows.py --duration 30 [--mock-latency-ms N]`

## 关键文件
- `submission/` — 提交包（client.py, run.sh, setup.sh, requirements.txt）
- `mock_vllm.py` — vLLM 替身
- `start_ubiservice_local.py` — Windows 友好的 ubiservice 启动器
- `run_eval_windows.py` — Windows 版 run_eval
- `tests/test_logprob_correctness.py` — logprob 数学正确性测试
