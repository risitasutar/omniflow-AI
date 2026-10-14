"""Agent tools for image generation and memory proposals.

image_generate is EXTERNAL_WRITE (the prompt may be sent to Cloudflare and a new file is
saved), so the graph pauses for an explicit approval of the exact prompt before it runs.
A real-provider failure is reported as an error - it never falls back to a mock image - and
mock output is always labelled "MOCK placeholder image (not AI-generated)".
memory_propose only proposes; the user confirms in the UI.

(Gmail and Google Calendar tools were removed on 2026-10-13 at the user's request.)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.integrations.base import IntegrationError
from omniflow.integrations.hub import IntegrationsHub
from omniflow.integrations.images import save_image, validate_prompt
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

MOCK_LABEL = "MOCK placeholder image (not AI-generated)"
REAL_LABEL = "Image generated with Cloudflare Workers AI (flux-1-schnell)"


class ImageArgs(BaseModel):
    """Arguments for image_generate."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(description="What the image should show (3-2048 characters).")
    steps: int = Field(default=4, ge=1, le=8, description="Diffusion steps (1-8).")


class MemoryProposalArgs(BaseModel):
    """Arguments for memory_propose."""

    model_config = ConfigDict(extra="forbid")
    content: str = Field(description="The durable fact or preference, in the user's terms.")
    category: str = Field(default="explicit", description="preference, project or explicit.")

    @field_validator("category")
    @classmethod
    def _c(cls, v: str) -> str:
        if v not in ("preference", "project", "explicit"):
            raise ValueError("category must be preference, project or explicit")
        return v


def make_integration_tools(hub: IntegrationsHub) -> list[ToolSpec]:
    def owner(context: ToolContext) -> str:
        return context.owner_id or "local"

    def call(fn):
        try:
            return fn()
        except IntegrationError as exc:
            raise ToolError(exc.category, exc.safe_message, retryable=False) from exc

    def image_generate(args: ImageArgs, context: ToolContext) -> dict:
        prompt = call(lambda: validate_prompt(args.prompt))
        provider = call(hub.image_provider)
        raw = call(lambda: provider.generate(prompt, args.steps))
        path = save_image(hub.settings.generated_images_dir, owner(context), raw,
                          provider.name)
        mock = provider.name == "mock"
        label = MOCK_LABEL if mock else REAL_LABEL
        return {"query": "generate image", "source_type": "action", "mock": mock,
                "image_path": str(path),
                "results": [{"id": 1, "title": label, "url": "",
                             "snippet": f"{label}; saved as {path.name}."}]}

    def memory_propose(args: MemoryProposalArgs, context: ToolContext) -> dict:
        from omniflow.memory.policy import MemoryRejected, assess

        try:
            assessed = assess(args.content)
        except MemoryRejected as exc:
            raise ToolError("rejected", str(exc)) from exc
        return {"query": "propose memory", "source_type": "action",
                "memory_proposal": {"content": assessed.content, "category": args.category,
                                    "sensitive": assessed.sensitive},
                "results": [{"id": 1, "title": "Memory proposal (not saved)", "url": "",
                             "snippet": f"Proposed to remember ({args.category}): "
                                        f"{assessed.content}. Nothing is saved until the user "
                                        "confirms."}]}

    def image_availability(context: ToolContext | None = None) -> Availability:
        info = hub.image_info()
        if info.usable:
            return Availability(AvailabilityStatus.AVAILABLE,
                                "MOCK placeholders" if info.status.value == "mock"
                                else info.detail)
        status = (AvailabilityStatus.MISSING_CREDENTIALS if info.status.value == "needs_setup"
                  else AvailabilityStatus.DISABLED)
        return Availability(status, info.detail)

    return [
        ToolSpec(
            name="image_generate",
            description="Generate an image from a text prompt with the configured provider. "
            "Requires the user's approval (the prompt may be sent to an external provider).",
            args_schema=ImageArgs, func=image_generate, category="image_generation",
            permission=Permission.EXTERNAL_WRITE, side_effect=SideEffect.REVERSIBLE,
            timeout_seconds=90.0, max_retries=0, uses_context=True,
            availability=image_availability,
            preview=lambda a: f"Generate an image\nPrompt: {a.get('prompt')}\nSteps: "
                              f"{a.get('steps', 4)}\nProvider: "
                              f"{hub.settings.image_generation_provider}"),
        ToolSpec(
            name="memory_propose",
            description="Propose saving a durable fact or preference to the user's memory. "
            "Only a proposal: the user must confirm. Use only when the user asks to remember "
            "something or clearly wants it kept.",
            args_schema=MemoryProposalArgs, func=memory_propose, category="memory",
            permission=Permission.LOCAL_READ, side_effect=SideEffect.NONE,
            timeout_seconds=10.0, max_retries=0, uses_context=True),
    ]
