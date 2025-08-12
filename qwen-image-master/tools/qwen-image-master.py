import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from loguru import logger

from tools.tool_client import ToolCallingClient, ToolCallingPayload


class QwenImageMasterTool(Tool):
    """https://www.modelscope.cn/docs/model-service/API-Inference/intro"""

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        logger.debug(f"{json.dumps(tool_parameters, indent=2, ensure_ascii=False)}")

        tp = ToolCallingPayload(**tool_parameters)

        tcc = ToolCallingClient(
            api_key=self.runtime.credentials["MODELSCOPE_API_KEY"],
            base_url=self.runtime.credentials.get(
                "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/"
            ),
        )

        if response := tcc.invoke(tp):
            json_message = response.model_dump(mode="json")
            logger.success(f"{json.dumps(json_message, indent=2, ensure_ascii=False)}")
            yield self.create_json_message(json_message)
            if image_url := response.image_url:
                yield self.create_image_message(image_url)
        else:
            logger.error("Image Generation Failed.")
