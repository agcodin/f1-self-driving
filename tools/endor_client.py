"""Typed client for one Endor decision call.

Wraps POST /v1/systemone with the behaviour the API actually needs: a timeout
long enough for a cold start, and retries that honour Retry-After.

The key is read from ENDOR_API_KEY, or from a gitignored .env beside the repo.
It is never written into a command line, where it would land in shell history
and in the process list.

    from endor_client import EndorClient, Choice, Noul
    client = EndorClient(model="f1training/decider-2b")
    ans = client.decide(
        {"speed_kmh": 312, "braking_demand_ahead": {"80m": 0.4}},
        {"pedal": Choice("Which input is fastest here?", {"BRAKE": "...", "FULL_POWER": "..."}),
         "confident": Noul("Will this turn out well?")},
    )
    ans.choice("pedal")   # -> ("FULL_POWER", 0.91)
    ans.noul("confident") # -> 0.83
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API_URL = "https://api.endor.thetokencompany.com/v1/systemone"

# Decision servers stop after 15 minutes idle and the next call restarts them.
# A cold start can take minutes, and the server gives up at 5, so allow 5.5.
COLD_START_TIMEOUT = 330.0
# Retried only for warming_up, rate_limited and overloaded, which are transient.
RETRY_STATUSES = {429, 503, 529}
MAX_RETRIES = 6
MAX_BACKOFF = 60.0
STATE_LIMIT_BYTES = 256 * 1024


class EndorError(RuntimeError):
    """A decision request failed."""

    def __init__(self, status: int, code: str | None, message: str, param: str | None = None):
        self.status, self.code, self.param = status, code, param
        super().__init__(f"{status} {code or 'error'}: {message}" + (f" (param {param})" if param else ""))


@dataclass(frozen=True)
class Choice:
    """Pick one of `criteria`, keyed by option id."""
    instructions: str
    criteria: dict[str, str | None]

    def wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is the probability of yes."""
    instructions: str
    criteria: dict[str, str] | None = None

    def wire(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = dict(self.criteria)
        return q


@dataclass(frozen=True)
class Score:
    """An ordered scale. `levels` runs lowest to highest, 2 to 10 of them."""
    instructions: str
    levels: list[str]

    def wire(self) -> dict[str, Any]:
        if not 2 <= len(self.levels) <= 10:
            raise ValueError(f"a score question needs 2 to 10 levels, got {len(self.levels)}")
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


Question = Choice | Noul | Score


@dataclass(frozen=True)
class Answers:
    """One decision response."""
    model: str
    raw: dict[str, Any]
    input_tokens: int = 0
    attempts: int = 1
    waited: float = 0.0

    def _answer(self, name: str, want: str) -> dict[str, Any]:
        a = self.raw.get("answers", {}).get(name)
        if a is None:
            raise KeyError(f"no answer for {name!r}; got {sorted(self.raw.get('answers', {}))}")
        if a.get("type") != want:
            raise TypeError(f"{name!r} is a {a.get('type')!r} question, not {want!r}")
        return a

    def noul(self, name: str) -> float:
        """Probability of yes."""
        return float(self._answer(name, "noul")["noul"])

    def choice(self, name: str) -> tuple[str, float]:
        """The chosen option id and its probability."""
        a = self._answer(name, "choice")
        probs = a.get("probs") or {}
        pick = a.get("choice")
        return pick, float(probs.get(pick, float("nan")))

    def probs(self, name: str) -> dict[str, float]:
        """Full distribution over a choice question's options."""
        return {k: float(v) for k, v in (self._answer(name, "choice").get("probs") or {}).items()}

    def score(self, name: str) -> float:
        return float(self._answer(name, "score")["score"])


def _load_env() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class EndorClient:
    model: str
    # repr=False on both: a dataclass repr lands in tracebacks and logs, and a
    # key printed there is a key leaked.
    api_key: str | None = field(default=None, repr=False)
    url: str = API_URL
    timeout: float = COLD_START_TIMEOUT
    max_retries: int = MAX_RETRIES
    # _key is deliberately NOT a dataclass field: dataclasses.asdict() walks
    # fields regardless of repr, so declaring it there would put the key into
    # anything that serialises the client.

    def __post_init__(self) -> None:
        if not self.api_key:
            _load_env()
        key = self.api_key or os.environ.get("ENDOR_API_KEY")
        if not key:
            raise EndorError(0, "no_api_key",
                             "set ENDOR_API_KEY, or put it in a gitignored .env beside the repo")
        self._key = key
        self.api_key = None   # keep exactly one copy, out of the repr
        if "/" not in self.model:
            raise ValueError(f"model must be '<project>/<id>', got {self.model!r} "
                             "(a bare base model id returns 422 model_requires_project)")

    def decide(self, state: Any, questions: dict[str, Question], *,
               exclude_from_training: bool = False) -> Answers:
        if not questions:
            raise ValueError("at least one question is required")
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {name: q.wire() for name, q in questions.items()},
        }
        if exclude_from_training:
            body["exclude_from_training"] = True

        payload = json.dumps(body).encode("utf-8")
        state_bytes = len(json.dumps(state).encode("utf-8"))
        if state_bytes > STATE_LIMIT_BYTES:
            raise EndorError(413, "payload_too_large",
                             f"state is {state_bytes} bytes, over the {STATE_LIMIT_BYTES} limit")

        waited, backoff = 0.0, 1.0
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(self.url, data=payload, method="POST", headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return Answers(model=data.get("model", self.model), raw=data,
                               input_tokens=int((data.get("usage") or {}).get("input_tokens", 0)),
                               attempts=attempt, waited=waited)
            except urllib.error.HTTPError as exc:
                code, message, param = _error_fields(exc)
                if exc.code not in RETRY_STATUSES or attempt == self.max_retries:
                    raise EndorError(exc.code, code, message, param) from None
                # warming_up and rate_limited both say when to come back.
                delay = _retry_after(exc.headers.get("Retry-After"), backoff)
                time.sleep(delay)
                waited += delay
                backoff = min(backoff * 2, MAX_BACKOFF)
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise EndorError(0, "connection_error", str(exc.reason)) from None
                time.sleep(backoff)
                waited += backoff
                backoff = min(backoff * 2, MAX_BACKOFF)
        raise EndorError(0, "unreachable", "retries exhausted")


def _error_fields(exc: urllib.error.HTTPError) -> tuple[str | None, str, str | None]:
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return None, exc.reason or "request failed", None
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    return err.get("code"), err.get("message") or exc.reason or "request failed", err.get("param")


def _retry_after(header: str | None, fallback: float) -> float:
    """Seconds to wait. Retry-After is seconds here, but tolerate junk."""
    if header:
        try:
            return max(0.0, min(float(header), MAX_BACKOFF))
        except ValueError:
            pass
    return fallback
