from abc import ABC, abstractmethod
from typing import Any
import httpx
from .config import settings

class InferenceAdapter(ABC):
    @abstractmethod
    async def health(self) -> dict[str, Any]: ...
    @abstractmethod
    async def generate(self, prompt: str) -> dict[str, Any]: ...

class DemoAdapter(InferenceAdapter):
    async def health(self):
        return {"provider":"demo","connected":False,"mode":"simulation","message":"No real model endpoint configured."}
    async def generate(self, prompt: str):
        return {"provider":"demo","mode":"simulation","text":"Demo only. Configure a real model endpoint to execute inference. Prompt received: "+prompt[:200]}

class NimAdapter(InferenceAdapter):
    async def health(self):
        return {"provider":"nvidia-nim","connected":False,"mode":"configured_not_verified","message":"Verify the selected NIM model API and health contract."}
    async def generate(self, prompt: str):
        raise NotImplementedError("Implement the exact selected NIM model API contract before enabling real inference.")

def get_adapter():
    if settings.nvidia_nim_base_url and settings.nvidia_nim_api_key:
        return NimAdapter()
    return DemoAdapter()
