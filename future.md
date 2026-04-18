# Future — 待完成的工作

## 真机验证清单（最高优先级，本地做不了）
- [ ] 4x5090 + Qwen3-32B BF16 真机：vLLM tp=4 能起来、显存占用、首任务延迟
- [ ] 监控 setup.sh 时间（pip install，<20min 限额）
- [ ] 监控 run.sh 启动时间（CLAUDE.md 说 60s 预算；FP8 在线量化可能额外 +30-60s）
- [ ] 真机跑一轮基线（QUANT_MODE=fp8 + draft_model 投机解码开）记录：
  - 每档位 pass% / avg / max 延迟
  - thinking_trunc 计数（观察 max_gen_toks 是否常被 thinking 吃光）
  - 实际总得分
- [ ] 对已知 prompt/continuation 做 logprob 手算对比（验证 mock 已通过但真实 Qwen3-32B tokenizer 无边界异常）
- [ ] 投机解码接受率统计（查 vLLM 日志）。<50% 就关 SPEC_MODEL；>70% 才真正赚

## 基于真机数据的决策
- [ ] 如果 Supreme/Glorious 100% LATE 且 GPU 满载：开 SKIP_SLA_TIERS=Supreme,Glorious
- [ ] 如果 thinking_trunc > 10%：试 SHORT_GEN_NOTHINK_THRESHOLD=256
- [ ] 如果 FP8 启动吃掉 run.sh 超出 60s 预算：QUANT_MODE=none 回退
- [ ] 如果 FP8 还不够快：跑 prepare_awq_model.py 下 AWQ，切 QUANT_MODE=awq
- [ ] 如果分低得多（AWQ 损失质量）：回 FP8 + 更激进的 vLLM 调参

## 中优先级
- [ ] 调优 vLLM 参数（max-num-seqs、max-num-batched-tokens、gpu_memory_utilization）的实际最优值
- [ ] 调优客户端并发参数（NUM_FETCHERS、MAX_INFLIGHT、QUERY_INTERVAL）
- [ ] 排查本地联调偶发 400 Bad Request（Memurai 并发语义 / 客户端真实 bug，真机上再看）
- [ ] NUM_SPEC_TOKENS 调优（默认 5，实测接受率决定）

## 低优先级
- [ ] 压力测试长时间稳定性
- [ ] 本地 Git Bash 验证 run.sh 执行链（保险起见）
- [ ] 考虑预量化 FP8 checkpoint 放 submission 减少启动量化时间（但体积大）
