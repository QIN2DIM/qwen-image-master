from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from pydantic import BaseModel


class ToolPayload(BaseModel):
    query: str
    parameter_generator: Any
    imagine_model: str


class QwenImageLazyTool(Tool):
    """https://www.modelscope.cn/docs/model-service/API-Inference/intro"""

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        yield self.create_json_message({"result": "Hello, world!"})
