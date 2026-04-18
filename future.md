# Future — 待完成的工作

## 高优先级
- [ ] 本地用 ubiservice 模拟服务器测试完整流程（不需要真实模型）
- [ ] 在真实 GPU 环境测试 vLLM 启动 + 推理
- [ ] 验证 loglikelihood 的 echo+logprobs 解析逻辑是否正确
- [ ] 验证 generate_until 的采样参数传递是否正确

## 中优先级
- [ ] 调优 vLLM 参数（batch size、max_model_len、gpu_memory_utilization）
- [ ] 调优客户端并发参数（fetcher 数量、inflight 上限）
- [ ] 考虑是否需要 apply_chat_template（影响正确性 vs 速度权衡）
- [ ] 测试极端 SLA（Supreme 0.5s）下是否应该跳过某些任务

## 低优先级
- [ ] 考虑是否引入投机解码（speculative decoding）
- [ ] 打包为 .tar.gz 并用 run_eval.py 做端到端测试
- [ ] 压力测试：长时间运行下的稳定性
