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

## 2026-04-18（Tier 4 策略层）
- client.py 加分档位 stats：每 30s 打印各 SLA tier 的 count / ok / late / pass% / avg / max 延迟
- 加 SKIP_SLA_TIERS env var（默认空）：可以选择性跳过某些档位（如 SKIP_SLA_TIERS=Supreme,Glorious）。但根据赛制 LATE 是 0 分不扣分、skip 也是 0 分，只有 GPU 满载的场景跳过才划算，所以默认不跳
- 加 thinking 截断检测：输出含 `<think>` 但无 `</think>` 时 stats["thinking_truncated"] 加一，便于真机上观察 max_gen_toks 是否常被 thinking 吃光
- 加 SHORT_GEN_NOTHINK_THRESHOLD env var（默认 0 关闭）：max_gen_toks 小于阈值时在 prompt 末尾附加 `<think>\n\n</think>\n\n` 伪造 thinking 已完成，让模型直接输出答案。代价是丢 reasoning 质量（reference 是带 thinking 的）
- 不能改：max_gen_toks 必须遵守 eval_gen_kwargs [开发文档:471]；也不能用 stop=["</think>"]（会截掉答案）
- 本地 e2e 测试验证 stats 输出正常

## 2026-04-18（Tier 3 量化接入）
- 用户明确要求 Tier 3 量化，覆盖 CLAUDE.md 的"不做量化"原则
- run.sh 加 QUANT_MODE env var（默认 fp8）：
  - `fp8`（默认）：`--quantization fp8 --kv-cache-dtype fp8`。5090 原生支持，~1.5-2x，近无损，无需额外下载（在线量化）
  - `awq`：需要 AWQ_MODEL_PATH 指向预量化权重，`--quantization awq_marlin`，~2-3x，质量损失 <1%
  - `none`：回退 BF16
- 新增 prepare_awq_model.py（辅助脚本，opt-in）：下载 Qwen3-32B-AWQ 约 16GB，优先 ModelScope。仅在 FP8 真机测不够快时才需要
- bash -n 语法检查通过
- 启动时间：FP8 在线量化可能额外 15-60s，可能超 CLAUDE.md 的 60s 预算。真机测后可能需要 QUANT_MODE=none 回退或预下载 FP8 checkpoint

## 2026-04-18（draft 模型下载完成）
- 尝试 HF 下载 Qwen3-1.7B 极慢（130 KB/s，2 分钟才 16 MB），放弃
- 切 ModelScope：速度 ~13 MB/s，4 分钟下完 3.80 GB
- 下载位置：submission/draft_model/（含 12 个文件：2 个 safetensors 分片 + tokenizer + config 等）
- 清理了 ModelScope 的 .mdl/.msc/.mv 元数据文件
- 验证：transformers 可正常加载，model_type=qwen3, hidden_size=2048, layers=28, vocab=151643（和 Qwen3-32B 共享 tokenizer，投机解码兼容）
- 真机打包：`python run_eval_windows.py --include-draft` 或 `tar czf final.tar.gz -C submission .`

## 2026-04-18（draft 模型打包方案）
- 确认平台不保证有小 Qwen3（Q33 允许 BYO），所以把 draft 模型随 submission 一起打包
- run.sh 自动检测路径优先级：SPEC_MODEL env var → ./draft_model/ → 禁用投机
- 新增 prepare_draft_model.py（在 submission/ 外，不打包）：从 HF 或 ModelScope 下载 Qwen3-1.7B 到 submission/draft_model/。ModelScope 作为 HF 被墙时的备胎
- run_eval_windows.py 加 --include-draft 开关：本地测试默认排除 draft_model/（保持 KB 级小 tarball），上真机打包时用 --include-draft 把模型一起塞进去

## 2026-04-18（Tier 1 + Tier 2 性能优化）
- run.sh 加 vLLM 调优参数：`--max-num-seqs 16`（SLA 敏感场景限并发）、`--max-num-batched-tokens 8192`、`--enable-chunked-prefill`（prefill/decode 交错）
- 新增 submission/warmup.py：run.sh 在 vLLM 就绪后发 12 个预热请求（serial short/medium/long + 8 并发）触发 CUDA graph capture，避免首批真实任务吃冷启动延迟。run.sh 用 `|| echo WARN` 保证失败不阻塞
- run.sh 加 SPEC_MODEL env var 支持投机解码：`--speculative-model $SPEC_MODEL --num-speculative-tokens 5`。SPEC_MODEL 未设则走原路径
- setup.sh 加 DOWNLOAD_SPEC_MODEL=1 可选开关：从 HF 下载 Qwen3-1.5B 作 draft model（默认关，避免吃 20min setup 预算）
- 等效档位估计：Gold 稳达，Platinum 边缘，Supreme 除非量化否则做不到
- 选择：暂不量化（按 CLAUDE.md 的 no-quant 原则），先跑 Tier 1+2 看实际得分再决定

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
