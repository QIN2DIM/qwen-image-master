# -*- coding: utf-8 -*-
"""
@Time    : 2025/8/13 01:58
@Author  : QIN2DIM
@GitHub  : https://github.com/QIN2DIM
@Desc    :
"""
import time
from typing import List

import httpx
from loguru import logger
from pydantic import BaseModel, Field


class ToolCallingPayload(BaseModel):
    prompt: str
    negative_prompt: str | None = Field(default="")
    size: str | None = Field(default="2048x2048")
    seed: int | None = Field(default=114514, ge=0, le=2147483647)
    steps: int | None = Field(default=30, ge=1, le=100)
    guidance: float | None = Field(default=3.5, ge=1.5, le=20)
    model: str | None = Field(default="Qwen/Qwen-Image")


class ToolCallingResponse(BaseModel):
    request_id: str | None = ""
    task_id: str | None = ""
    task_status: str | None = ""
    time_taken: float | None = 0.0
    input: dict | None = Field(default_factory=dict)
    output_images: List[str] | None = Field(default_factory=list)

    @property
    def image_url(self) -> str:
        return self.output_images[0]


class ToolCallingClient:
    def __init__(self, api_key: str, base_url: str = "https://api-inference.modelscope.cn/"):
        self._api_key = api_key
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url)

        self._loop_count = 60
        self._sleep_interval = 3

    def invoke(self, tp: ToolCallingPayload) -> ToolCallingResponse | None:
        common_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = tp.model_dump(mode="json")
        response = self._client.post(
            "/v1/images/generations",
            json=payload,
            headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
        )
        response.raise_for_status()

        task_id = response.json()["task_id"]

        for _ in range(self._loop_count):
            result = self._client.get(
                f"/v1/tasks/{task_id}",
                headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
            )
            result.raise_for_status()
            data = result.json()

            logger.debug(f"{data=}")

            if data.get("task_status") in ["SUCCEED", "FAILED"]:
                return ToolCallingResponse(**data)

            time.sleep(self._sleep_interval)
