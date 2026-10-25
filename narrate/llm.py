"""LLM client, in order of preference:
  1. Azure OpenAI / Microsoft Foundry deployment (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT, key or Entra ID)
  2. GitHub Models - free, rate-limited, only needs a GitHub token (GITHUB_MODELS_TOKEN, or GITHUB_TOKEN with HUNT_LLM=github)
  3. Any OpenAI-compatible endpoint (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
None when unconfigured: the system then runs fully on deterministic templates.

Env:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION (default 2024-10-21)
  AZURE_OPENAI_API_KEY  - or omit it to use Entra ID (managed identity) via azure-identity
  GITHUB_MODELS_TOKEN, GITHUB_MODELS_MODEL (default openai/gpt-4.1-mini), GITHUB_MODELS_ENDPOINT
  OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL - generic fallback
"""
from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


class LLM:
    def __init__(self, client, model: str, provider: str):
        self.client, self.model, self.provider = client, model, provider

    def chat(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 400) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()


@lru_cache(maxsize=1)
def get_llm() -> LLM | None:
    if os.getenv("HUNT_AUTOPSY_OFFLINE"):
        return None
    try:
        from openai import AzureOpenAI, OpenAI
    except Exception:  # pragma: no cover
        return None
    ep, dep = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_DEPLOYMENT")
    ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if ep and dep:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if key:
            return LLM(AzureOpenAI(azure_endpoint=ep, api_key=key, api_version=ver), dep, "azure-openai")
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            tp = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            return LLM(AzureOpenAI(azure_endpoint=ep, azure_ad_token_provider=tp, api_version=ver), dep, "azure-openai-entra")
        except Exception:
            return None
    gh = os.getenv("GITHUB_MODELS_TOKEN") or (os.getenv("GITHUB_TOKEN") if os.getenv("HUNT_LLM") == "github" else None)
    if gh:
        ep = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference")
        return LLM(OpenAI(base_url=ep, api_key=gh), os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini"), "github-models")
    if os.getenv("OPENAI_API_KEY"):
        return LLM(OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None), os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "openai")
    return None
