from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag.rag_qa import RAGQAService
from app.rag.schemas import QAResponse
from app.langchain.langchain_rag_chain import LangChainRAGService
from app.langgraph.langgraph_rag_graph import LangGraphRAGService


app = FastAPI(
    title="Agent RAG QA API",
    version="0.5.0",
    description="基于本地知识库的多轮 RAG 问答接口（含上下文压缩与分层记忆调试）",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QARequest(BaseModel):
    query: str = Field(..., description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回的 top-k 条数")
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于维护多轮对话上下文"
    )


class ClearSessionRequest(BaseModel):
    session_id: str = Field(..., description="需要清空的会话ID")


@lru_cache(maxsize=1)
def get_service() -> RAGQAService:
    return RAGQAService()

@lru_cache(maxsize=1)
def get_langchain_service() -> LangChainRAGService:
    return LangChainRAGService()

@lru_cache(maxsize=1)
def get_langgraph_service() -> LangGraphRAGService:
    return LangGraphRAGService()


@app.get("/")
def root() -> dict:
    return {"message": "RAG QA API is running."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/qa/langgraph", response_model=QAResponse)
def qa_langgraph(request: QARequest) -> QAResponse:
    try:
        service = get_langgraph_service()
        return service.ask(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/qa/langchain", response_model=QAResponse)
def qa_langchain(request: QARequest) -> QAResponse:
    try:
        service = get_langchain_service()
        return service.ask(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/qa", response_model=QAResponse)
def qa(request: QARequest) -> QAResponse:
    try:
        service = get_service()
        return service.ask(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/session/clear")
def clear_session(request: ClearSessionRequest) -> dict:
    try:
        service = get_service()
        service.clear_session_history(request.session_id)
        return {
            "message": f"会话 {request.session_id} 的历史记录已清空。"
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/debug")
def session_debug(
    session_id: str = Query(..., description="需要查看的会话ID")
) -> dict:
    try:
        service = get_service()
        return service.get_session_debug_info(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memory/list")
def memory_list() -> dict:
    try:
        service = get_service()
        memories = service.list_long_term_memories()
        return {
            "count": len(memories),
            "memories": memories,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/memory/clear")
def memory_clear() -> dict:
    try:
        service = get_service()
        service.clear_long_term_memories()
        return {"message": "长期记忆已清空。"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memory/debug")
def memory_debug(
    query: str = Query(..., description="用于测试长期记忆召回的query")
) -> dict:
    try:
        service = get_service()
        return service.get_memory_debug_info(query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc