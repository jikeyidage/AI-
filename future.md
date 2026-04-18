# Future — 待完成的工作

## 高优先级（真实环境才能做）
- [ ] 4x5090 + Qwen3-32B 真机：验证 vLLM 启动、tp=4 tensor parallel、显存占用
- [ ] 真机验证 logprob 数学：用已知 prompt/continuation，对比 client 返回的 logprob 与手算值（虽然 mock 已通过，但真实 tokenizer 行为可能有细节差异）
- [ ] 真机验证 generate_until：检查采样参数（temperature/top_p/top_k）是否正确传递到 vLLM，以及输出是否包含 thinking 标签（应该会）
- [ ] 用 ubiservice/bin/run_eval.py（Unix 版）在真机上做完整端到端评测
- [ ] 监控 setup.sh 的 pip 下载时间，确认在 20 分钟限额内；监控 run.sh 拉起 vLLM 的时间（Qwen3-32B BF16 加载）

## 中优先级
- [ ] 调优 vLLM 参数（max-num-seqs、max-num-batched-tokens、gpu_memory_utilization）
- [ ] 调优客户端并发参数（NUM_FETCHERS、MAX_INFLIGHT、QUERY_INTERVAL）
- [ ] 极端 SLA（Supreme 0.5s）下是否应该跳过某些任务 —— 本地压测显示 tight SLA 下会 LATE 但仍提交，符合赛制预期（LATE 只是不得分，不扣分，比未提交好）
- [ ] 排查本地联调中出现的偶发 400 Bad Request（可能是 Memurai 并发语义问题，也可能是真实客户端 bug）
- [ ] max_gen_toks 不够时 thinking 没出完答案的应对策略（[QA.txt:398] 是风险点）

## 低优先级
- [ ] 投机解码（speculative decoding）
- [ ] 压力测试长时间稳定性
- [ ] FP8 / INT8 量化（如果显存吃紧）
- [ ] 本地可选：如果想纯 Windows 验证 bash 执行链，可以装 Git Bash 跑 run.sh
