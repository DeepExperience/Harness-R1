# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Client for accessing GenRM service.

This module provides an async client to communicate with the GenRM service for
generative reward model evaluations.
"""

from typing import Dict, List, Optional

import httpx

from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)

# Module-level singleton to avoid creating a new client per request
_genrm_client: Optional["GenRMClient"] = None


class GenRMClient:
    """Async client for GenRM service.

    Provides async methods to interact with the GenRM service for
    reward/preference evaluations. Uses httpx.AsyncClient to avoid blocking the
    event loop in async rollout contexts.
    """

    def __init__(self, service_url: Optional[str] = None, timeout: float = 1800.0):
        """Initialize GenRM client.

        Args:
            service_url: URL of the GenRM service. If None, will use get_serve_url("genrm")
            timeout: Request timeout in seconds
        """
        if service_url is None:
            service_url = get_serve_url("genrm")

        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self._async_client = httpx.AsyncClient(timeout=timeout)
        # Keep a sync client for health checks during init and non-async contexts
        self._sync_client = httpx.Client(timeout=timeout)

        logger.info(f"GenRMClient initialized with service URL: {self.service_url}")

    async def generate(
        self,
        messages: List[dict],
        sampling_params: Optional[Dict] = None,
    ) -> str:
        """Async generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to GenRM service,
        and returns the raw model response. The caller is responsible
        for formatting the messages and parsing the response.

        Args:
            messages: List of messages with role and content
                (e.g., [{"role": "user", "content": "..."}])
            sampling_params: Optional sampling parameters to override defaults.
                Supported keys: temperature, top_p, top_k, max_new_tokens.
                Example: {"temperature": 0.3, "top_p": 0.9}

        Returns:
            Raw response string from the GenRM model
        """
        url = f"{self.service_url}/generate"
        payload: Dict = {
            "messages": messages,
        }
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params

        try:
            resp = await self._async_client.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            # Return the raw response string
            return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"GenRM generate request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in GenRM generate: {e}")
            raise

    def health_check(self) -> Dict:
        """Check health status of GenRM service (sync, safe to call at init).

        Returns:
            Dictionary with status information
        """
        url = f"{self.service_url}/health"
        try:
            resp = self._sync_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_metrics(self) -> Dict:
        """Get metrics from GenRM service.

        Returns:
            Dictionary with service metrics
        """
        url = f"{self.service_url}/metrics"
        try:
            resp = self._sync_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM metrics request failed: {e}")
            return {}

    def close(self):
        """Close both HTTP clients."""
        self._sync_client.close()
        try:
            self._async_client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_genrm_client(service_url: Optional[str] = None, timeout: float = 1800.0) -> GenRMClient:
    """Get or create a singleton GenRM client.

    Uses module-level caching to avoid creating a new HTTP client and
    health-check round trip on every call.

    Args:
        service_url: URL of the GenRM service
        timeout: Request timeout in seconds

    Returns:
        GenRMClient instance (cached singleton)
    """
    global _genrm_client
    if _genrm_client is None:
        _genrm_client = GenRMClient(service_url=service_url, timeout=timeout)
    return _genrm_client
