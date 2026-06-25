from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionCatalogItem:
    action_id: str
    action_type: str
    title: str
    description: str
    target_step: str | None
    base_weight: float
    primary_metric: str
    expected_impact: str
    execution_hint: dict[str, str | int | float | bool | None] = field(default_factory=dict)


ACTION_CATALOG: dict[str, ActionCatalogItem] = {
    "emphasize_reviews": ActionCatalogItem(
        action_id="emphasize_reviews",
        action_type="CONTENT",
        title="리뷰/구매후기 영역 강조",
        description="상품 상세 화면에서 리뷰와 구매후기 영역을 더 눈에 띄게 노출합니다.",
        target_step="product_view_to_add_to_cart",
        base_weight=0.85,
        primary_metric="view_to_cart_rate",
        expected_impact="상품 신뢰도를 높여 장바구니 전환율 개선을 기대합니다.",
    ),
    "show_price_benefit": ActionCatalogItem(
        action_id="show_price_benefit",
        action_type="CONTENT",
        title="가격 혜택/할인 정보 노출 강화",
        description="상품 상세와 주요 CTA 주변에 할인, 적립, 배송 혜택 정보를 강화합니다.",
        target_step="product_view_to_add_to_cart",
        base_weight=0.80,
        primary_metric="view_to_cart_rate",
        expected_impact="구매 전 가격 저항을 낮춰 장바구니 추가를 유도합니다.",
    ),
    "improve_product_detail": ActionCatalogItem(
        action_id="improve_product_detail",
        action_type="LANDING",
        title="상품 상세 정보 개선",
        description="상품 상세 설명, 이미지, 핵심 구매 포인트를 보강합니다.",
        target_step="product_view_to_add_to_cart",
        base_weight=0.75,
        primary_metric="view_to_cart_rate",
        expected_impact="상품 이해도를 높여 상세 조회 이후 이탈을 줄입니다.",
    ),
    "cart_reminder_message": ActionCatalogItem(
        action_id="cart_reminder_message",
        action_type="MESSAGE",
        title="장바구니 리마인드 메시지 발송",
        description="장바구니에 상품을 담고 결제를 시작하지 않은 사용자에게 리마인드를 보냅니다.",
        target_step="add_to_cart_to_checkout_start",
        base_weight=0.80,
        primary_metric="cart_to_checkout_rate",
        expected_impact="장바구니 방치 사용자의 결제 시작 전환을 높입니다.",
    ),
    "limited_time_coupon": ActionCatalogItem(
        action_id="limited_time_coupon",
        action_type="INCENTIVE",
        title="제한 시간 쿠폰 제공",
        description="결제 시작을 유도하기 위해 짧은 유효기간의 쿠폰 혜택을 제공합니다.",
        target_step="add_to_cart_to_checkout_start",
        base_weight=0.76,
        primary_metric="cart_to_checkout_rate",
        expected_impact="즉시 결제 동기를 높여 장바구니 이후 이탈을 줄입니다.",
        execution_hint={"coupon_type": "limited_time"},
    ),
    "free_shipping_coupon": ActionCatalogItem(
        action_id="free_shipping_coupon",
        action_type="INCENTIVE",
        title="무료배송 쿠폰 제공",
        description="결제 직전 이탈 사용자를 대상으로 무료배송 쿠폰을 테스트합니다.",
        target_step="checkout_start_to_purchase",
        base_weight=0.84,
        primary_metric="checkout_to_purchase_rate",
        expected_impact="배송비 저항을 낮춰 결제 완료율 개선을 기대합니다.",
        execution_hint={"coupon_type": "free_shipping"},
    ),
    "checkout_reminder_message": ActionCatalogItem(
        action_id="checkout_reminder_message",
        action_type="MESSAGE",
        title="결제 완료 리마인드 메시지 발송",
        description="결제를 시작했지만 구매를 완료하지 않은 사용자에게 리마인드를 보냅니다.",
        target_step="checkout_start_to_purchase",
        base_weight=0.78,
        primary_metric="checkout_to_purchase_rate",
        expected_impact="결제 중단 사용자를 다시 유입시켜 구매 완료를 유도합니다.",
    ),
    "payment_friction_check": ActionCatalogItem(
        action_id="payment_friction_check",
        action_type="UX_CHECK",
        title="결제 UX/오류 지점 점검",
        description="결제 단계의 오류, 로딩, 입력 불편, 결제수단 문제를 점검합니다.",
        target_step="checkout_start_to_purchase",
        base_weight=0.72,
        primary_metric="checkout_to_purchase_rate",
        expected_impact="결제 과정의 마찰을 줄여 구매 완료율을 개선합니다.",
    ),
    "recommend_alternative_product": ActionCatalogItem(
        action_id="recommend_alternative_product",
        action_type="PRODUCT",
        title="대체 상품 추천",
        description="품절 상품을 조회한 사용자에게 유사하거나 대체 가능한 상품을 추천합니다.",
        target_step=None,
        base_weight=0.86,
        primary_metric="view_to_purchase_rate",
        expected_impact="품절로 인한 구매 손실을 대체 상품 전환으로 보완합니다.",
    ),
    "restock_notification": ActionCatalogItem(
        action_id="restock_notification",
        action_type="MESSAGE",
        title="재입고 알림 신청 유도",
        description="품절 상품 상세에서 재입고 알림 신청을 유도합니다.",
        target_step=None,
        base_weight=0.74,
        primary_metric="return_visit_rate",
        expected_impact="재입고 시 재방문과 구매 가능성을 높입니다.",
    ),
    "pause_out_of_stock_ads": ActionCatalogItem(
        action_id="pause_out_of_stock_ads",
        action_type="AD",
        title="품절 상품 광고 일시 중단",
        description="품절 상품으로 유입되는 광고 집행을 일시 중단하거나 대체 상품으로 전환합니다.",
        target_step=None,
        base_weight=0.88,
        primary_metric="ad_spend_efficiency",
        expected_impact="전환 불가능한 광고비 낭비를 줄입니다.",
    ),
    "adjust_landing_page": ActionCatalogItem(
        action_id="adjust_landing_page",
        action_type="LANDING",
        title="채널별 랜딩 페이지 변경",
        description="전환이 낮은 유입 채널의 랜딩 페이지를 세그먼트 의도에 맞게 조정합니다.",
        target_step=None,
        base_weight=0.78,
        primary_metric="view_to_purchase_rate",
        expected_impact="유입 의도와 랜딩 경험을 맞춰 구매 전환을 개선합니다.",
    ),
    "revise_ad_message": ActionCatalogItem(
        action_id="revise_ad_message",
        action_type="AD",
        title="광고 메시지 문구 수정",
        description="전환이 낮은 채널 또는 캠페인의 광고 문구와 혜택 메시지를 수정합니다.",
        target_step=None,
        base_weight=0.76,
        primary_metric="view_to_cart_rate",
        expected_impact="광고 기대와 실제 랜딩 경험의 차이를 줄입니다.",
    ),
    "adjust_ad_targeting": ActionCatalogItem(
        action_id="adjust_ad_targeting",
        action_type="AD",
        title="광고 타겟 조정",
        description="전환이 낮은 채널 또는 캠페인의 타겟 조건을 재조정합니다.",
        target_step=None,
        base_weight=0.74,
        primary_metric="view_to_purchase_rate",
        expected_impact="광고 예산을 전환 가능성이 높은 사용자에게 집중합니다.",
    ),
    "send_reminder_without_coupon": ActionCatalogItem(
        action_id="send_reminder_without_coupon",
        action_type="MESSAGE",
        title="쿠폰 없이 리마인드만 발송",
        description="이미 구매 가능성이 높은 사용자에게 비용성 혜택 없이 리마인드만 발송합니다.",
        target_step=None,
        base_weight=0.82,
        primary_metric="purchase_rate",
        expected_impact="전환은 유지하면서 쿠폰 비용을 줄입니다.",
    ),
    "exclude_coupon_target": ActionCatalogItem(
        action_id="exclude_coupon_target",
        action_type="COST_CONTROL",
        title="쿠폰 지급 대상에서 제외",
        description="쿠폰 없이도 구매 가능성이 높은 사용자를 쿠폰 지급 대상에서 제외합니다.",
        target_step=None,
        base_weight=0.80,
        primary_metric="coupon_cost_per_purchase",
        expected_impact="불필요한 쿠폰 비용과 마진 훼손을 줄입니다.",
    ),
    "manual_review": ActionCatalogItem(
        action_id="manual_review",
        action_type="REVIEW",
        title="운영자 수동 검토 필요",
        description="현재 원인 후보에 직접 매칭되는 액션 카탈로그가 없어 수동 검토가 필요합니다.",
        target_step=None,
        base_weight=0.50,
        primary_metric="purchase_rate",
        expected_impact="운영자가 원인 후보를 직접 검토해 적절한 후속 액션을 결정합니다.",
    ),
}

VIEW_TO_CART_ACTION_IDS = [
    "emphasize_reviews",
    "show_price_benefit",
    "improve_product_detail",
]
CART_TO_CHECKOUT_ACTION_IDS = [
    "cart_reminder_message",
    "limited_time_coupon",
]
CHECKOUT_TO_PURCHASE_ACTION_IDS = [
    "free_shipping_coupon",
    "checkout_reminder_message",
    "payment_friction_check",
]
OUT_OF_STOCK_ACTION_IDS = [
    "recommend_alternative_product",
    "restock_notification",
    "pause_out_of_stock_ads",
]
CHANNEL_CONVERSION_ACTION_IDS = [
    "adjust_landing_page",
    "revise_ad_message",
    "adjust_ad_targeting",
]
HIGH_PURCHASE_INTENT_ACTION_IDS = [
    "send_reminder_without_coupon",
    "exclude_coupon_target",
]
