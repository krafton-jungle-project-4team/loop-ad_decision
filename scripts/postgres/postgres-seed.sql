-- postgres-seed.sql
-- 데모/초기 데이터 insert 전용

-- =========================================================
-- 1. Demo Project
-- =========================================================

INSERT INTO projects (
    id,
    name,
    domain,
    sdk_key,
    status
)
VALUES (
    'google-ga4-demo-commerce',
    'GA4 Demo Commerce',
    'demo-shop.loop-ad.local',
    'demo-sdk-key-google-ga4-commerce',
    'active'
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    domain = EXCLUDED.domain,
    sdk_key = EXCLUDED.sdk_key,
    status = EXCLUDED.status,
    updated_at = now();

-- =========================================================
-- 2. Dashboard User
-- 데모용. 실제 서비스에서는 password_hash를 제대로 생성해야 한다.
-- =========================================================

INSERT INTO dashboard_users (
    project_id,
    email,
    password_hash,
    role,
    status
)
VALUES (
    'google-ga4-demo-commerce',
    'admin@loop-ad.local',
    NULL,
    'admin',
    'active'
)
ON CONFLICT (project_id, email) DO UPDATE SET
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    updated_at = now();

-- =========================================================
-- 3. Automation Policy
-- 자동 실행 정책 데모값
-- 쿠폰은 차단하고, PRODUCT/CONTENT/AD만 자동 실행 허용
-- =========================================================

INSERT INTO automation_policies (
    project_id,
    enabled,
    auto_execute_enabled,
    allowed_action_ids,
    allowed_action_types,
    blocked_action_ids,
    max_experiment_traffic_ratio,
    min_priority_score,
    max_discount_rate,
    max_daily_coupon_budget,
    max_message_per_user_per_day,
    stop_loss_relative_drop
)
VALUES (
    'google-ga4-demo-commerce',
    true,
    true,
    '[]'::jsonb,
    '["PRODUCT", "CONTENT", "AD"]'::jsonb,
    '["limited_time_coupon", "free_shipping_coupon"]'::jsonb,
    0.2,
    0.6,
    0.1,
    1000000,
    1,
    0.05
)
ON CONFLICT (project_id) DO UPDATE SET
    enabled = EXCLUDED.enabled,
    auto_execute_enabled = EXCLUDED.auto_execute_enabled,
    allowed_action_ids = EXCLUDED.allowed_action_ids,
    allowed_action_types = EXCLUDED.allowed_action_types,
    blocked_action_ids = EXCLUDED.blocked_action_ids,
    max_experiment_traffic_ratio = EXCLUDED.max_experiment_traffic_ratio,
    min_priority_score = EXCLUDED.min_priority_score,
    max_discount_rate = EXCLUDED.max_discount_rate,
    max_daily_coupon_budget = EXCLUDED.max_daily_coupon_budget,
    max_message_per_user_per_day = EXCLUDED.max_message_per_user_per_day,
    stop_loss_relative_drop = EXCLUDED.stop_loss_relative_drop,
    updated_at = now();

-- =========================================================
-- 4. Action Catalog
-- =========================================================

INSERT INTO action_catalog (
    action_id,
    action_type,
    title,
    description,
    target_step,
    base_weight,
    primary_metric,
    expected_impact,
    execution_hint_json,
    status
)
VALUES
(
    'recommend_alternative_product',
    'PRODUCT',
    '대체 상품 추천',
    '품절 상품을 조회한 사용자에게 대체 상품을 추천합니다.',
    NULL,
    0.86,
    'view_to_purchase_rate',
    '품절로 인한 구매 손실을 대체 상품 전환으로 보완합니다.',
    '{}'::jsonb,
    'active'
),
(
    'pause_out_of_stock_ads',
    'AD',
    '품절 상품 광고 일시 중단',
    '품절 상품으로 유입되는 광고를 일시 중단하거나 대체 상품으로 전환합니다.',
    NULL,
    0.88,
    'ad_spend_efficiency',
    '전환 불가능한 광고비 낭비를 줄입니다.',
    '{}'::jsonb,
    'active'
),
(
    'show_price_benefit',
    'CONTENT',
    '가격 혜택 노출 강화',
    '상품 상세와 CTA 주변에 할인, 적립, 배송 혜택 정보를 강화합니다.',
    'product_view_to_add_to_cart',
    0.80,
    'view_to_cart_rate',
    '가격 저항을 낮춰 장바구니 전환율 개선을 기대합니다.',
    '{}'::jsonb,
    'active'
),
(
    'limited_time_coupon',
    'INCENTIVE',
    '제한 시간 쿠폰 제공',
    '짧은 유효기간의 쿠폰으로 결제 시작을 유도합니다.',
    'add_to_cart_to_checkout_start',
    0.76,
    'cart_to_checkout_rate',
    '즉시 결제 동기를 높입니다.',
    '{"coupon_type": "limited_time", "discount_rate": 0.1}'::jsonb,
    'active'
),
(
    'manual_review',
    'REVIEW',
    '운영자 수동 검토 필요',
    '자동 매칭 가능한 액션이 없어 운영자 검토가 필요합니다.',
    NULL,
    0.50,
    'purchase_rate',
    '운영자가 원인을 직접 확인하고 후속 액션을 결정합니다.',
    '{}'::jsonb,
    'active'
)
ON CONFLICT (action_id) DO UPDATE SET
    action_type = EXCLUDED.action_type,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    target_step = EXCLUDED.target_step,
    base_weight = EXCLUDED.base_weight,
    primary_metric = EXCLUDED.primary_metric,
    expected_impact = EXCLUDED.expected_impact,
    execution_hint_json = EXCLUDED.execution_hint_json,
    status = EXCLUDED.status,
    updated_at = now();

-- =========================================================
-- 5. Campaigns
-- =========================================================

INSERT INTO campaigns (
    project_id,
    external_campaign_id,
    name,
    channel,
    goal,
    budget,
    status,
    metadata_json
)
VALUES
(
    'google-ga4-demo-commerce',
    'campaign_fresh_google_001',
    '신선식품 구글 유입 캠페인',
    'google',
    'purchase',
    1000000,
    'active',
    '{}'::jsonb
),
(
    'google-ga4-demo-commerce',
    'campaign_fresh_onsite_001',
    '신선식품 온사이트 추천 캠페인',
    'onsite',
    'purchase',
    500000,
    'active',
    '{}'::jsonb
)
ON CONFLICT (project_id, external_campaign_id) DO UPDATE SET
    name = EXCLUDED.name,
    channel = EXCLUDED.channel,
    goal = EXCLUDED.goal,
    budget = EXCLUDED.budget,
    status = EXCLUDED.status,
    metadata_json = EXCLUDED.metadata_json,
    updated_at = now();

-- =========================================================
-- 6. Coupons
-- =========================================================

INSERT INTO coupons (
    project_id,
    code,
    name,
    discount_type,
    discount_rate,
    discount_amount,
    max_discount_amount,
    budget,
    status,
    metadata_json
)
VALUES
(
    'google-ga4-demo-commerce',
    'LIMITED10',
    '제한 시간 10% 쿠폰',
    'percentage',
    0.1000,
    NULL,
    5000,
    1000000,
    'active',
    '{}'::jsonb
)
ON CONFLICT (project_id, code) DO UPDATE SET
    name = EXCLUDED.name,
    discount_type = EXCLUDED.discount_type,
    discount_rate = EXCLUDED.discount_rate,
    discount_amount = EXCLUDED.discount_amount,
    max_discount_amount = EXCLUDED.max_discount_amount,
    budget = EXCLUDED.budget,
    status = EXCLUDED.status,
    metadata_json = EXCLUDED.metadata_json,
    updated_at = now();

-- =========================================================
-- 7. Ad Creatives
-- campaign_id는 subquery로 연결
-- =========================================================

INSERT INTO ad_creatives (
    project_id,
    campaign_id,
    coupon_id,
    action_id,
    creative_type,
    title,
    message,
    image_url,
    landing_url,
    payload_json,
    status
)
VALUES
(
    'google-ga4-demo-commerce',
    (
        SELECT id FROM campaigns
        WHERE project_id = 'google-ga4-demo-commerce'
          AND external_campaign_id = 'campaign_fresh_onsite_001'
    ),
    NULL,
    'recommend_alternative_product',
    'banner',
    '품절 상품 대신 이 상품은 어떠세요?',
    '비슷한 인기 상품을 추천드려요.',
    NULL,
    '/products/alternative-fresh-001',
    '{"placement": "product_detail"}'::jsonb,
    'active'
),
(
    'google-ga4-demo-commerce',
    (
        SELECT id FROM campaigns
        WHERE project_id = 'google-ga4-demo-commerce'
          AND external_campaign_id = 'campaign_fresh_google_001'
    ),
    NULL,
    'pause_out_of_stock_ads',
    'admin_action',
    '품절 상품 광고 일시 중단',
    '품절 상품 광고를 일시 중단합니다.',
    NULL,
    NULL,
    '{"admin_action": true}'::jsonb,
    'active'
),
(
    'google-ga4-demo-commerce',
    (
        SELECT id FROM campaigns
        WHERE project_id = 'google-ga4-demo-commerce'
          AND external_campaign_id = 'campaign_fresh_onsite_001'
    ),
    (
        SELECT id FROM coupons
        WHERE project_id = 'google-ga4-demo-commerce'
          AND code = 'LIMITED10'
    ),
    'limited_time_coupon',
    'banner',
    '지금 구매하면 10% 할인',
    '짧은 시간 동안만 제공되는 쿠폰입니다.',
    NULL,
    '/coupon/LIMITED10',
    '{"placement": "cart"}'::jsonb,
    'active'
);