# Past — 已完成的工作

## 2026-04-18
- 阅读并分析了赛制介绍、开发文档、QA（115条）
- 阅读了 ubiservice 本地模拟服务器全部源码
- 确定了整体技术方案：vLLM (tp=4) + 异步客户端调度
- 确定了关键技术决策：不用 chat template、不显式关闭 thinking mode、不量化
- 创建了 CLAUDE.md 和项目状态跟踪文件（past/now/future）
- 编写了 submission/ 目录下所有文件：
  - setup.sh: pip install vllm httpx transformers torch
  - run.sh: 启动 vLLM (tp=4, prefix-caching, 8192 max-model-len) + client.py
  - client.py: 异步调度客户端，支持 generate_until / loglikelihood / loglikelihood_rolling
  - requirements.txt: 依赖列表

## 2026-04-18（正确性审查 + 压力测试）
- 通读 QA.txt (115条) + 开发文档 (474行) + 赛制介绍 (124行)，关键事实：
  - [QA.txt:293, 322, 326] prompt 是原始文本，reference 答案基于原始文本通过 /v1/completions 生成 → 不用 chat template 决策正确
  - [QA.txt:191, 192, 218] loglikelihood 和 loglikelihood_rolling 提交"总 logprob"
  - [QA.txt:249, 291] Qwen3-32B thinking mode 默认开启（reference 生成时也开）
  - [QA.txt:190, 赛制41] 未提交 -2× reward；空串/固定串 → 恶意提交额外惩罚
  - [QA.txt:185] 初赛最多 64 个同时 ask 未 submit
  - [QA.txt:331] 实际数据集上 \n 不是常见停止符，不用担心 thinking 被截断
  - [QA.txt:398] max_gen_toks 不够导致答案没出完 → 0 分（风险点）
- 发现并修复 logprob 数学 bug：原代码用 tokenizer.encode(prompt) 和 tokenizer.encode(full_text) 分别编码后切片，BPE tokenizer 在边界会合并 token 导致切片错位。改为"分别 encode 后拼成 token_ids 列表，作为 prompt 字段传给 vLLM（vLLM 支持 integer list 输入，不会 re-tokenize）"。精确可靠
- 写 tests/test_logprob_correctness.py + mock_vllm 加 --deterministic 模式：每 token logprob=-1.0 生成 token=-2.0，验证 7+2 个边界 case 全通过
- 写 run_eval_windows.py（Windows 版 run_eval.py）：打包 submission → 解压到 _eval_work → 设 run_eval.py 风格的环境变量 → 跑 ubiservice + mock_vllm + client → 收 leaderboard
- 端到端测试（30s）：108 任务 / 0 LATE / 98.6ms 延迟
- SLA 压力测试：3s mock 延迟 vs Gold 6s → 105 任务 / 0 LATE；7s 延迟 vs Gold 6s → 全部 LATE（预期）
- 修 bug：emergency submit 原来不检查 HTTP 状态码，400 也会打印"succeeded"。加了 raise_for_status
- 打包时排除 __pycache__ 和 .pyc（tarball 从 16.5KB 缩到 6.1KB）
- Emergency submit 的 generate_until 默认从 "" 改为 "."（避免被判为空串恶意提交）
- 本地联调有过 400 Bad Request（submit 时偶发）根因未定位，可能是 Memurai 下 Redis 并发语义问题，也可能是客户端真实 bug。真机验证前保持警惕

## 2026-04-18（本地联调）
- 通读 submission 代码，发现并修复 7 个问题：
  - QUERY_INTERVAL 0.05 → 0.2（4 fetcher × 20qps 满足 matcher 32/s 限流）
  - 加 tokenizer fallback（HF 下载失败 / 本地无模型时用近似 tokenizer）
  - run.sh 加 `--served-model-name "$MODEL_NAME"`（让 vLLM 和 client 用同一个模型名）
  - run.sh 移除 `--enforce-eager`（恢复 CUDA graphs 加速）
  - fetcher 改为先 acquire 信号量再 /ask（避免 /ask 到不能立刻处理的任务，浪费 SLA）
  - 删除 fetcher 的 dead param `task_queue`
- 创建 inference conda 环境（Python 3.11），装 fastapi/uvicorn/redis/httpx/requests/transformers（uvloop Windows 不支持已跳过）
- 装 Memurai Developer Edition 作为 Windows Redis 替代，服务已跑起来
- 写了 mock_vllm.py：FastAPI 实现 /v1/models 和 /v1/completions（支持 echo/logprobs 两种模式），供本地无 GPU 时代替真 vLLM
- 写了 start_ubiservice_local.py：Windows 友好启动器，跳过 ubiservice 自带的 redis-server/redis-cli 调用（直接复用已运行的 Memurai）
- 本地端到端联调跑通：15 秒内完成 149 任务，0 个 LATE/ERROR，延迟 0.09-0.18s（Gold SLA=6s）
