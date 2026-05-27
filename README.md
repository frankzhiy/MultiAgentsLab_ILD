# MultiAgentsLab_ILD

**面向 ILD MDT 的机制驱动 LLM 多智能体研究平台**

本项目旨在构建一个以机制设计为核心的多智能体协作研究框架，聚焦间质性肺疾病（ILD）多学科诊疗团队（MDT）场景，探索 LLM 驱动的智能体如何通过结构化机制（信息共享、冲突检测、意见修订、仲裁决策）实现高质量的协作诊断推理。

---

## 项目目录结构

```
MultiAgentsLab_ILD/
├── configs/            # 配置文件（智能体、模型、工作流、共享机制、安全、评估）
├── data/               # 数据资产（原始病例、处理后病例、标注、示例、基准集）
├── docs/               # 文档（架构设计、设计笔记、阶段计划、数据模式、协议、评估）
├── notebooks/          # Jupyter Notebooks（探索性分析、结果分析）
├── scripts/            # 脚本工具（环境搭建、运行入口、验证、导出）
├── src/                      # 核心Python 包
│       ├── schemas/          # Pydantic 数据模型定义
│       ├── state/            # 全局/局部状态管理
│       ├── agents/           # 各专科智能体实现
│       ├── mechanisms/       # 核心机制模块
│       │   ├── sharing/      # 信息共享机制
│       │   ├── conflict/     # 冲突检测机制
│       │   ├── revision/     # 意见修订机制
│       │   └── arbitration/  # 仲裁决策机制
│       ├── validators/       # 输出验证器
│       ├── workflows/        # LangGraph 工作流编排
│       ├── prompts/          # Prompt 模板管理
│       ├── llm/              # LLM 客户端封装
│       ├── storage/          # 持久化存储接口
│       ├── evaluation/       # 评估指标与流程
│       └── utils/            # 通用工具函数
├── tests/              # 单元测试与集成测试
├── outputs/            # 运行输出（traces、报告、导出文件）
├── logs/               # 运行日志
└── .github/            # GitHub 配置与 Copilot 约束
```

---

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key 等配置
```

## 技术栈

- **Python** >= 3.11
- **LangGraph** — 多智能体工作流编排
- **LangChain** — LLM 抽象与工具链
- **Pydantic** — 数据模型与验证
- **OpenAI API** — 大语言模型接口
- **Ruff + MyPy** — 代码质量工具
- **Pytest** — 测试框架

---

## License

MIT License © Frank
