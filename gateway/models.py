from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerateRequest:
    prompt: str
    model: str
    max_tokens: int = 100
    temperature: float = 0.0


@dataclass
class GenerateResponse:
    text: str
    model: str
    provider: str
    latency_seconds: float
    output_tokens: int = 0
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)