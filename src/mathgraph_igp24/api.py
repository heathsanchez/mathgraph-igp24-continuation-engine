from __future__ import annotations

import getpass
import os
import time
from typing import Any, Sequence

from .models import Polynomial
from .polynomial import poly_to_line


def sanitize_api_key(raw: str) -> str:
    return "".join(character for character in raw.strip() if 32 < ord(character) < 127 and not character.isspace())


def get_api_key(prompt: bool = True) -> str:
    key = sanitize_api_key(os.environ.get("SAIR_API_KEY", ""))
    if not key and prompt: key = sanitize_api_key(getpass.getpass("SAIR API key: "))
    return key


class SairClient:
    def __init__(self, api_key: str, competition_id: str = "igp24", base_url: str = "https://api.sair.foundation/api/public/v1"):
        self.api_key = sanitize_api_key(api_key)
        if not self.api_key: raise ValueError("SAIR API key is required")
        self.competition_id = competition_id; self.base_url = base_url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _requests(self):
        try: import requests
        except ImportError as exc: raise RuntimeError("install requests to use SairClient") from exc
        return requests

    def _data(self, response):
        response.raise_for_status(); envelope = response.json(); return envelope.get("data", envelope)

    def competition(self) -> dict[str, Any]:
        return self._data(self._requests().get(f"{self.base_url}/competitions/{self.competition_id}", headers=self.headers, timeout=180))

    def eligibility(self) -> dict[str, Any]:
        return self._data(self._requests().get(f"{self.base_url}/competitions/{self.competition_id}/me", headers=self.headers, timeout=180))

    def submit(self, polynomials: Sequence[Polynomial], description: str = "MathGraph v102 provenance lawbook cycle") -> dict[str, Any]:
        eligibility = self.eligibility()
        if eligibility.get("canSubmit") is False: raise RuntimeError(str(eligibility.get("submitBlockedReason") or "submission is blocked"))
        payload = {"payload": {"polynomials": [poly_to_line(poly) for poly in polynomials]}, "meta": {"description": description[:500]}}
        response = self._requests().post(f"{self.base_url}/competitions/{self.competition_id}/submissions", headers=self.headers, json=payload, timeout=180)
        if response.status_code == 429: raise RuntimeError(f"rate limited: retry after {response.headers.get('Retry-After')}")
        return self._data(response)

    def submission(self, submission_id: str) -> dict[str, Any]:
        return self._data(self._requests().get(f"{self.base_url}/competitions/{self.competition_id}/submissions/{submission_id}", headers=self.headers, timeout=180))

    def poll(self, submission_id: str, max_polls: int = 160, seconds: int = 30) -> dict[str, Any]:
        latest = {}
        for _ in range(max_polls):
            latest = self.submission(submission_id)
            queued = (latest.get("payload", {}) or {}).get("queuedPolynomials", []) or []
            if not queued: return latest
            time.sleep(seconds)
        return latest

