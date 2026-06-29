from dataclasses import dataclass

from fastapi import status


@dataclass
class ContentGenerationError(Exception):
    code: str
    message: str
    status_code: int = status.HTTP_400_BAD_REQUEST

    def to_response_body(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


def not_found(code: str, message: str) -> ContentGenerationError:
    return ContentGenerationError(
        code=code,
        message=message,
        status_code=status.HTTP_404_NOT_FOUND,
    )
