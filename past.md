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
