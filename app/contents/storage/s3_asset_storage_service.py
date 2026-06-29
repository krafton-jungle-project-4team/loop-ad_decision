from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from app.core.config import Settings


@dataclass(frozen=True)
class StorageObject:
    key: str
    body: bytes
    content_type: str


class AssetStorage(Protocol):
    def public_url_for_key(self, key: str) -> str:
        raise NotImplementedError

    def upload_objects(self, objects: list[StorageObject]) -> None:
        raise NotImplementedError


class S3AssetStorage:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.loopad_data_storage_bucket
        base_url = settings.loopad_public_asset_base_url
        self.public_base_url = base_url.rstrip("/") if base_url else None

    def public_url_for_key(self, key: str) -> str:
        quoted_key = quote(key, safe="/")
        if self.public_base_url:
            return f"{self.public_base_url}/{quoted_key}"
        return f"https://{self.bucket}.s3.amazonaws.com/{quoted_key}"

    def upload_objects(self, objects: list[StorageObject]) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 asset uploads") from exc

        client = boto3.client("s3")
        for item in objects:
            client.put_object(
                Bucket=self.bucket,
                Key=item.key,
                Body=item.body,
                ContentType=item.content_type,
            )
