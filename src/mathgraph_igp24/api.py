from __future__ import annotations

from typing import Any, Sequence

from .models import Polynomial
from .polynomial import poly_to_line


class SairClient:
    def __init__(
        self,
        api_key: str,
        competition_id: str = "igp24",
        base_url: str = "https://api.sair.foundation/api/public/v1",
    ):
        if not api_key.strip():
            raise ValueError("SAIR API key is required")
        self.api_key = api_key.strip()
        self.competition_id = competition_id
        self.base_url = base_url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _requests(self):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("install requests to use SairClient") from exc
        return requests

    def competition(self) -> dict[str, Any]:
        response = self._requests().get(
            f"{self.base_url}/competitions/{self.competition_id}", headers=self.headers, timeout=180
        )
        response.raise_for_status()
        envelope = response.json()
        return envelope.get("data", envelope)

    def eligibility(self) -> dict[str, Any]:
        response = self._requests().get(
            f"{self.base_url}/competitions/{self.competition_id}/me", headers=self.headers, timeout=180
        )
        response.raise_for_status()
        envelope = response.json()
        return envelope.get("data", envelope)

    def submit(self, polynomials: Sequence[Polynomial], description: str = "MathGraph continuation engine") -> dict[str, Any]:
        eligibility = self.eligibility()
        if eligibility.get("canSubmit") is False:
            raise RuntimeError(str(eligibility.get("submitBlockedReason") or "submission is blocked"))
        payload = {
            "payload": {"polynomials": [poly_to_line(poly) for poly in polynomials]},
            "meta": {"description": description[:500]},
        }
        response = self._requests().post(
            f"{self.base_url}/competitions/{self.competition_id}/submissions",
            headers=self.headers, json=payload, timeout=180,
        )
        response.raise_for_status()
        envelope = response.json()
        return envelope.get("data", envelope)

    def submission(self, submission_id: str) -> dict[str, Any]:
        response = self._requests().get(
            f"{self.base_url}/competitions/{self.competition_id}/submissions/{submission_id}",
            headers=self.headers, timeout=180,
        )
        response.raise_for_status()
        envelope = response.json()
        return envelope.get("data", envelope)

