from typing import Any

from app.contents.ai.image_provider import GeneratedImage
from app.contents.compose.png_canvas import create_mock_background_png


class MockImageProvider:
    provider_name = "mock"

    def generate_background(self, brief: dict[str, Any]) -> GeneratedImage:
        return GeneratedImage(
            body=create_mock_background_png(width=1200, height=628),
            content_type="image/png",
            provider_name=self.provider_name,
            model=None,
        )
