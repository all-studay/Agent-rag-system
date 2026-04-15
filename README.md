# Agent RAG System

一个面向企业知识问答场景的 RAG 原型系统，支持文档读取、混合检索、结构化问答、多轮上下文管理、长期记忆，以及 LangChain / LangGraph 适配，适合作为校招、面试和作品集项目。

## 1. 项目简介

本项目实现了一个企业知识助理系统，能够对 FAQ、制度文档、入职手册等资料进行读取、切分、索引、检索与问答。在基础 RAG 的基础上，进一步加入了多轮对话、历史摘要压缩和长期记忆注入能力，并补充了 LangChain 与 LangGraph 版本的问答流程实现。

## 2. 核心功能

- 支持 TXT、Markdown、PDF 文档读取
- 支持文本切分、向量化与 Chroma 向量存储
- 支持向量检索 + BM25 的混合召回
- 支持 reranker 二次精排
- 支持结构化问答输出与 no-answer 检测
- 支持多轮对话、历史摘要压缩与长期记忆注入
- 支持 LangChain 链式封装与 LangGraph 状态流编排

## 3. 技术栈

**后端：**
- Python
- FastAPI
- Pydantic

**检索与模型：**
- sentence-transformers
- Chroma
- rank-bm25
- BAAI/bge-small-zh-v1.5
- BAAI/bge-reranker-base
- OpenAI Compatible API / Qwen

**框架适配：**
- LangChain
- LangGraph

**前端：**
- React
- TypeScript
- Vite
- Tailwind CSS
- Axios

## 4. 项目结构

```text
agent-rag-system/
├── app/
│   ├── api/
│   ├── chat/
│   ├── memory/
│   ├── rag/
│   ├── retrieval/
│   ├── langchain/
│   └── langgraph/
├── data/
│   ├── raw/
│   ├── vectorstore/
│   ├── memory/
│   └── eval/
├── frontend/
├── scripts/
├── .env
└── README.md
```

## 5. 快速启动
**安装后端依赖**
```Bash
conda create -n agent-rag python=3.11 -y
conda activate agent-rag
pip install fastapi uvicorn pydantic python-dotenv openai
pip install sentence-transformers chromadb rank-bm25 pypdf
pip install langchain langchain-core langchain-openai langgraph
```
**构建索引**
```Bash
python scripts/build_index.py
```

**启动后端**
```Bash
uvicorn app.api.main:app --reload
```

**启动前端**
```Bash
cd frontend
npm install
npm run dev
```

## 6. 项目亮点
        完成了从文档读取到问答输出的完整 RAG 闭环
        实现了混合检索与 reranker 精排，提升检索稳定性
        构建了 no-answer 检测机制，降低无依据回答风险
        实现多轮上下文管理与历史摘要压缩
        实现长期记忆抽取、召回与 Prompt 注入
        在原生实现基础上补充了 LangChain 与 LangGraph 版本
        提供前端 Demo 和评估脚本，便于演示与优化
## 7. 当前效果与后续方向
        当前项目已经完成基础工程原型，能够稳定支持企业知识问答、多轮对话与长期记忆增强，并具备前后端展示能力。在小规模测试集中，检索评估、问答评估和长期记忆评估均取得了较好的结果。
###后续可继续优化的方向包括：
        长期记忆向量化检索
        更完善的记忆更新与衰减策略
        更复杂的多智能体协作流程
        更完整的前端展示与产品化能力
