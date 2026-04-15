from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    """
        OpenAI 兼容接口客户端。
    """

    def __init__(
            self,
            model_name: str = os.getenv("LLM_MODEL_NAME", "qwen3.5-flash"),
            temperature: float = 0.2,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("未找到 OPENAI_API_KEY，请先在 .env 中配置。")

        client_kwargs = {"api_key" : api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.api_key: str = api_key
        self.base_url: str | None = base_url

    def generate(self, prompt: str) -> str:
        """
            调用大模型生成回答。
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "你是一个严谨的企业知识库问答助手。"},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content.strip()
