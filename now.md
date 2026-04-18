# Now — 现在在做什么

## 当前任务
- Tier 1/2/3/4 全部接入。等待真机验证。

## run.sh 调优开关（env var 控制）
- `QUANT_MODE=fp8|awq|none`（默认 fp8，Tier 3）
- `KV_CACHE_DTYPE=auto|fp8`（默认跟随 QUANT_MODE）
- `SPEC_MODEL=/path` 或自动检测 ./draft_model/（Tier 2）
- `NUM_SPEC_TOKENS=5`（默认 5）
- `AWQ_MODEL_PATH=/path`（QUANT_MODE=awq 时必需）
- `SKIP_SLA_TIERS=Supreme,Glorious`（Tier 4，默认空不跳过）
- `SHORT_GEN_NOTHINK_THRESHOLD=256`（Tier 4，默认 0 关闭，小 max_gen_toks 时跳过 thinking）

## 档位估计（待真机校准）
- Gold (6s)：稳达（主力档位）
- Platinum (4s)：边缘，靠 Tier 1/2 可能推稳
- Diamond (2s) 及以下：无量化不可行
- Supreme (0.5s)：必须量化（FP8 / AWQ）

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
