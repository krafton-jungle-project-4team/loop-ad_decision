from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedImage:
    body: bytes
    content_type: str
    provider_name: str
    model: str | None = None


class ImageProvider(Protocol):
    provider_name: str

    def generate_background(self, brief: dict[str, Any]) -> GeneratedImage:
        raise NotImplementedError
