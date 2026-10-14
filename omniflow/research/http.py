"""A locked-down HTTP GET for research providers.

Only code-defined hosts can be contacted - never a URL chosen by a user or the model.
Every request is HTTPS, has a timeout and a response-size cap, refuses to follow
redirects to other hosts, and refuses hosts that resolve to private, loopback,
link-local or cloud-metadata addresses.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import httpx

USER_AGENT = "OmniFlowAI/0.5 (personal research assistant; low-volume, cached requests)"
MAX_REDIRECTS = 2


class FetchError(Exception):
    """A fetch failed; `category` drives retry decisions and the safe message. `status` is
    the HTTP status when the provider answered with an error (None otherwise)."""

    def __init__(self, category: str, message: str, status: int | None = None):
        super().__init__(message)
        self.category = category
        self.status = status


_NAT64 = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052 well-known NAT64 prefix


def effective_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address):
    """The address traffic really goes to: NAT64 / IPv4-mapped IPv6 wrap an IPv4 address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in _NAT64:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


def address_is_public(address: str) -> bool:
    ip = effective_address(ipaddress.ip_address(address))
    # is_global excludes private, loopback, link-local (incl. 169.254.169.254 cloud
    # metadata), carrier-grade NAT, reserved and documentation ranges.
    return ip.is_global and not ip.is_multicast


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    return bool(infos) and all(address_is_public(info[4][0]) for info in infos)


def check_url(url: str, allowed_hosts: frozenset[str]) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https":
        raise FetchError("unsafe_url", "Only HTTPS sources are allowed.")
    if host not in allowed_hosts:
        raise FetchError("unsafe_url", f"'{host}' is not an allowed source.")
    if parts.username or parts.password or (parts.port not in (None, 443)):
        raise FetchError("unsafe_url", "URLs with credentials or custom ports are not allowed.")
    return host


def safe_get(
    url: str,
    *,
    allowed_hosts: frozenset[str],
    params: dict | None = None,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
    resolve_check=_host_is_public,
) -> tuple[bytes, dict[str, str]]:
    """Fetch an allowlisted HTTPS URL; return (body, lower-cased headers)."""
    current = url
    with httpx.Client(timeout=timeout, follow_redirects=False,
                      headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            host = check_url(current, allowed_hosts)
            if not resolve_check(host):
                raise FetchError("unsafe_url", f"'{host}' does not resolve to a public address.")
            try:
                with client.stream("GET", current, params=params) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location", "")
                        current = str(httpx.URL(current).join(location))
                        params = None  # the redirect target carries its own query
                        continue
                    if response.status_code == 429:
                        raise FetchError("rate_limited", "The provider's rate limit was reached.",
                                         429)
                    if response.status_code >= 500:
                        raise FetchError("provider_error",
                                         "The provider is temporarily unavailable.",
                                         response.status_code)
                    if response.status_code >= 400:
                        status = response.status_code
                        raise FetchError("provider_error",
                                         f"The provider rejected the request ({status}).", status)
                    declared = int(response.headers.get("content-length") or 0)
                    if declared > max_bytes:
                        raise FetchError("too_large", "The provider's response is too large.")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise FetchError("too_large", "The provider's response is too large.")
                    headers = {k.lower(): v for k, v in response.headers.items()}
                    return bytes(body), headers
            except httpx.TimeoutException as exc:
                raise FetchError("timeout", "The provider did not respond in time.") from exc
            except httpx.TransportError as exc:
                raise FetchError("network", "The provider could not be reached.") from exc
    raise FetchError("unsafe_url", "Too many redirects.")
