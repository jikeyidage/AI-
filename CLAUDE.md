# AI 推理挑战赛项目

## 项目概述
参加 Ubiquant Challenge AI 推理挑战赛（初赛）。需要在 4x RTX 5090 32GB 上部署 Qwen3-32B 推理服务，从平台拉取动态任务流，在 SLA 时限内完成推理并提交结果，最大化总得分。

## 状态跟踪
每次启动时请阅读以下文件了解项目进度：
- [past.md](past.md) — 已完成的工作
- [now.md](now.md) — 当前正在做的事情
- [future.md](future.md) — 待完成的工作

## 关键技术决策
- 推理引擎：vLLM，tensor parallel=4
- 模型：Qwen3-32B 全精度 (BF16)，**不做量化**
- Thinking mode：不显式关闭（与参考答案生成方式一致）
- Chat template：**不使用**。prompt 是原始文本，reference results 也基于原始文本生成
- Prompt 长度上限：4096 tokens；max_gen_toks 上限：4000
- max_model_len：8192
- 初赛无竞争机制，accept 任务必定成功

## 关键规则摘要（来自 QA）
- SLA 时间 = 从 /ask 返回到 /submit 提交的总时间（含所有 messages）
- 同一 task 内所有 messages 类型一致（正式比赛）
- loglikelihood/loglikelihood_rolling 提交**总 logprob**
- generate_until 的 response 需要**去掉停止符**
- 第一次 submit 结果有效，重复提交无效
- 未提交惩罚 = -2 x target_reward，必须避免
- generate_until 评分是语义级别（提取正确答案），不是序列相似度
- 恶意提交（连续错误）会影响 score 和信誉分
- setup.sh 限时 20 分钟，下载速度百兆级
- run.sh 需 60 秒内拉起推理服务
- 总运行时间：若干小时
- 4 张 5090 之间是 PCIe 互联，无 NVLink
- 使用conda 名字为inference的环境进行开发
