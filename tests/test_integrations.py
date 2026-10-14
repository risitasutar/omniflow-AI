"""Image-generation integration: HTTP helper, provider selection, Cloudflare client, mocks.

All HTTP goes through httpx.MockTransport - no network, no real credentials.
"""

import base64
import io

import httpx
import pytest
from PIL import Image

from omniflow.core.config import Settings
from omniflow.integrations import images
from omniflow.integrations.base import IntegrationError, Status, request_json, safe_url
from omniflow.integrations.hub import IntegrationsHub

ACCOUNT = "7e61b2719a544ed32fb9cc753f6f0000"  # fake 32-hex account id
TOKEN = "cf-FAKE-token-abcdefghijklmnop"


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "orange").save(buffer, format="JPEG")
    return buffer.getvalue()


# ---------------------------------------------------------------- HTTP helper
@pytest.mark.parametrize("status,category", [(401, "auth_expired"), (403, "permission"),
                                             (404, "not_found"), (400, "invalid_request"),
                                             (500, "unavailable")])
def test_http_errors_become_safe_categories(status, category, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = transport(lambda r: httpx.Response(status, json={"error": "secret detail"}))
    with pytest.raises(IntegrationError) as err:
        request_json(client, "GET", "https://example.org/x", token=TOKEN)
    assert err.value.category == category and "secret detail" not in err.value.safe_message


def test_log_urls_mask_account_ids_and_queries():
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/x?token=abc"
    masked = safe_url(url)
    assert ACCOUNT not in masked and "token=abc" not in masked
    assert masked.startswith("https://api.cloudflare.com/client/v4/accounts/…")


# --------------------------------------------------------- provider selection
def _hub(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return IntegrationsHub(Settings(_env_file=None),
                           http=transport(lambda r: httpx.Response(500)))


def test_default_is_disabled(monkeypatch):
    hub = _hub(monkeypatch)
    assert hub.image_info().status is Status.DISABLED
    with pytest.raises(IntegrationError):
        hub.image_provider()


def test_root_cause_regression_old_global_mock_setting_cannot_force_the_mock(monkeypatch):
    """The bug: OMNIFLOW_INTEGRATIONS_MODE=mock overrode OMNIFLOW_IMAGE_PROVIDER=cloudflare."""
    hub = _hub(monkeypatch, OMNIFLOW_INTEGRATIONS_MODE="mock",
               OMNIFLOW_IMAGE_PROVIDER="cloudflare", CLOUDFLARE_ACCOUNT_ID=ACCOUNT,
               CLOUDFLARE_API_TOKEN=TOKEN)
    assert hub.image_info().status is Status.CONNECTED
    assert isinstance(hub.image_provider(), images.CloudflareImages)


def test_cloudflare_without_credentials_needs_setup(monkeypatch):
    hub = _hub(monkeypatch, OMNIFLOW_IMAGE_PROVIDER="cloudflare")
    info = hub.image_info()
    assert info.status is Status.NEEDS_SETUP and "CLOUDFLARE_API_TOKEN" in info.detail


def test_explicit_mock_provider(monkeypatch):
    hub = _hub(monkeypatch, OMNIFLOW_IMAGE_PROVIDER="mock")
    assert hub.image_info().status is Status.MOCK
    assert isinstance(hub.image_provider(), images.MockImages)


# --------------------------------------------------------- Cloudflare client
def test_cloudflare_success_decodes_validates_and_uses_the_documented_endpoint():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"success": True, "result": {
            "image": base64.b64encode(jpeg_bytes()).decode()}})

    raw = images.CloudflareImages(ACCOUNT, TOKEN, transport(handler)).generate("a cat", 4)
    assert raw == jpeg_bytes()
    request = seen[0]
    assert request.url.path == (f"/client/v4/accounts/{ACCOUNT}/ai/run/"
                                "@cf/black-forest-labs/flux-1-schnell")
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.content == b'{"prompt":"a cat","steps":4}'


@pytest.mark.parametrize("status,phrase", [
    (401, "rejected the API token"), (403, "Workers AI permission"),
    (404, "CLOUDFLARE_ACCOUNT_ID"), (429, "free daily Workers AI allocation"),
    (500, "HTTP 500")])
def test_cloudflare_errors_are_specific_and_include_safe_codes(status, phrase):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"success": False, "errors": [
            {"code": 10000, "message": f"Authentication error for account {ACCOUNT}"}]})

    with pytest.raises(IntegrationError) as err:
        images.CloudflareImages(ACCOUNT, TOKEN, transport(handler)).generate("a cat", 4)
    message = err.value.safe_message
    assert phrase in message and "error code 10000" in message
    assert "No image was generated" in message
    assert ACCOUNT not in message and TOKEN not in message
    assert len(calls) == 1  # never retried
    assert "Reconnect" not in message  # the old, Google-specific advice


def test_cloudflare_timeout_and_bad_images_are_errors():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(IntegrationError, match="60 seconds"):
        images.CloudflareImages(ACCOUNT, TOKEN, transport(slow)).generate("a cat", 4)
    broken = transport(lambda r: httpx.Response(200, json={"success": True,
                                                           "result": {"image": "bm90"}}))
    with pytest.raises(IntegrationError, match="couldn't be read"):
        images.CloudflareImages(ACCOUNT, TOKEN, broken).generate("a cat", 4)
    empty = transport(lambda r: httpx.Response(200, json={"success": False, "errors": [
        {"code": 5006}], "result": None}))
    with pytest.raises(IntegrationError, match="error code 5006"):
        images.CloudflareImages(ACCOUNT, TOKEN, empty).generate("a cat", 4)


# --------------------------------------------------------- mock and storage
def test_mock_images_are_labelled_and_never_overwrite(tmp_path):
    raw = images.MockImages().generate("a red bicycle", 4)
    assert Image.open(io.BytesIO(raw)).size == (512, 512)
    first = images.save_image(tmp_path, "local", raw, "mock")
    second = images.save_image(tmp_path, "local", raw, "mock")
    assert first != second and first.exists() and second.exists()
    assert "mock" in first.name
    with pytest.raises(IntegrationError):
        images.validate_prompt("x")


def test_google_integrations_are_gone():
    import importlib.util

    for module in ("omniflow.integrations.gmail", "omniflow.integrations.calendar",
                   "omniflow.integrations.google_auth"):
        assert importlib.util.find_spec(module) is None
    settings = Settings(_env_file=None)
    assert not any(name.startswith(("google_oauth", "integrations_"))
                   for name in type(settings).model_fields)
