"""The external integration OmniFlow still has: image generation.

The provider is chosen ONLY by OMNIFLOW_IMAGE_PROVIDER:
- "disabled" (default): no image generation;
- "mock": a placeholder stamped "MOCK - not AI-generated" (development and tests only);
- "cloudflare": Cloudflare Workers AI with the user's CLOUDFLARE_ACCOUNT_ID /
  CLOUDFLARE_API_TOKEN.
No other setting can switch a configured real provider to the mock, and a failing real
provider never falls back to the mock: the user sees the provider's error instead.

(Gmail and Google Calendar integrations were removed on 2026-10-13 at the user's request.)
"""

from __future__ import annotations

import httpx

from omniflow.core.config import Settings
from omniflow.integrations import images
from omniflow.integrations.base import IntegrationError, IntegrationInfo, Status

CLOUDFLARE_SETUP_URL = "https://dash.cloudflare.com/profile/api-tokens"
ENV_VARS = ("OMNIFLOW_IMAGE_PROVIDER", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")


class IntegrationsHub:
    def __init__(self, settings: Settings, *, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self.http = http or httpx.Client()

    def image_info(self) -> IntegrationInfo:
        provider = self.settings.image_generation_provider
        perms = ("Sends your prompt to Cloudflare Workers AI (flux-1-schnell)",)
        if provider == "disabled":
            return IntegrationInfo("images", "Image generation", Status.DISABLED,
                                   "Disabled (OMNIFLOW_IMAGE_PROVIDER=disabled). Gemini image "
                                   "output has no free tier, so it isn't used.", ENV_VARS,
                                   CLOUDFLARE_SETUP_URL, perms)
        if provider == "mock":
            return IntegrationInfo("images", "Image generation", Status.MOCK,
                                   "MOCK placeholder images - not AI-generated "
                                   "(OMNIFLOW_IMAGE_PROVIDER=mock).", ENV_VARS,
                                   CLOUDFLARE_SETUP_URL, ())
        if not (self.settings.cloudflare_account_id and self.settings.cloudflare_api_token):
            return IntegrationInfo("images", "Image generation", Status.NEEDS_SETUP,
                                   "OMNIFLOW_IMAGE_PROVIDER=cloudflare, but CLOUDFLARE_ACCOUNT_ID "
                                   "or CLOUDFLARE_API_TOKEN is missing from .env.", ENV_VARS,
                                   CLOUDFLARE_SETUP_URL, perms)
        return IntegrationInfo("images", "Image generation", Status.CONNECTED,
                               "Cloudflare Workers AI is configured (credentials are checked "
                               "when the first image is generated).", ENV_VARS,
                               CLOUDFLARE_SETUP_URL, perms)

    def statuses(self, owner: str | None = None) -> list[IntegrationInfo]:
        return [self.image_info()]

    def info(self, key: str, owner: str | None = None) -> IntegrationInfo:
        if key != "images":
            raise KeyError(key)
        return self.image_info()

    def image_provider(self):
        info = self.image_info()
        if not info.usable:
            raise IntegrationError(info.status.value, f"Image generation isn't available: "
                                   f"{info.detail}")
        if info.status is Status.MOCK:
            return images.MockImages()
        return images.CloudflareImages(self.settings.cloudflare_account_id,
                                       self.settings.cloudflare_api_token.get_secret_value(),
                                       self.http)
