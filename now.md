# Now — 现在在做什么

## 当前任务
- 两处关键真机问题已修，等重新上传验证：
  1. run.sh CLI flags：`--no-enable-log-requests` + `--speculative-config` JSON（vLLM 0.9+ 语法）
  2. requirements.txt 精确版本化 + setup.sh 改为 `--no-deps vllm==0.9.2`，**不覆盖平台 Blackwell torch**
- 上平台后第一时间看两类日志：
  - setup：Step 1 的 `torch {ver}  cuda {ver}` 打印 → 确认 cu128；Step 4 import sanity → 确认 vllm 正常加载
  - run：`[run] SUCCESS: $SUCCESS_DESC` 落在 L1 / L1b / L3 / L4 哪档

## 当前提交包版本
- `vllm==0.9.2`（via `--no-deps`，保护平台 Blackwell torch）
- `httpx==0.27.2`
- 其余 vLLM 传递依赖（torch / transformers / tokenizers / xformers / ray / triton / ...）全部信任平台预装

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
- `submission/requirements.txt` — `vllm==0.9.2` + `httpx==0.27.2`，详细注释 --no-deps 策略
- `submission/setup.sh` — 5 步安装：verify torch → install vllm (--no-deps, skip if 0.9+) → install httpx → import check → optional draft download
- `submission/run.sh` — 四档 fallback，已全部用 0.9+ CLI flag
- `mock_vllm.py` — vLLM 替身
- `start_ubiservice_local.py` — Windows 友好的 ubiservice 启动器
- `run_eval_windows.py` — Windows 版 run_eval
- `tests/test_logprob_correctness.py` — logprob 数学正确性测试
