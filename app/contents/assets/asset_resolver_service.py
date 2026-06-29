from app.contents.assets.brand_asset_service import BrandAssetService
from app.contents.assets.product_asset_service import ProductAssetService


class AssetResolverService:
    def __init__(
        self,
        product_assets: ProductAssetService | None = None,
        brand_assets: BrandAssetService | None = None,
    ) -> None:
        self.product_assets = product_assets or ProductAssetService()
        self.brand_assets = brand_assets or BrandAssetService()

    def resolve_assets(self) -> dict[str, dict[str, str]]:
        return {
            "product": self.product_assets.resolve_product_layer(),
            "brand": self.brand_assets.resolve_brand_layer(),
        }
