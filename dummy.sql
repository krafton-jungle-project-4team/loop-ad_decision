-- =========================================================
-- Loop-Ad AI Decision Dummy / Seed Data
-- =========================================================
--
-- This file is intentionally separated from schema.sql.
-- Run schema.sql first, then dummy.sql.
--
-- dummy.sql includes:
--   1. demo-shop project
--   2. action_catalog seed
--   3. default segment/content/mapping fallback
--   4. sample normal segment with metrics only
--   5. sample anomalous segment with recommendation/content/experiment/mapping
--
-- It is designed to be idempotent where practical.
-- Running it multiple times should not create duplicate default serving rules.
--
-- =========================================================

-- =========================================================
-- 1. Demo project
-- =========================================================

INSERT INTO projects (project_key, name, timezone)
VALUES ('demo-shop', 'Demo Shopping Mall', 'Asia/Seoul')
ON CONFLICT (project_key) DO UPDATE
SET
    name = EXCLUDED.name,
    timezone = EXCLUDED.timezone;

-- =========================================================
-- 2. Action catalog
-- =========================================================

INSERT INTO action_catalog (
    action_key,
    name,
    description,
    target_funnel_step,
    default_channel,
    template_json
)
VALUES
(
    'highlight_benefit_banner',
    '혜택 강조 배너',
    '상품 상세 또는 리스트에서 할인, 무료배송, 리뷰 등 구매 유도 요소를 강조한다.',
    'view_to_cart',
    'banner',
    '{"content_type": "banner", "cta_label": "혜택 보기"}'::jsonb
),
(
    'cart_coupon_banner',
    '장바구니 쿠폰 배너',
    '장바구니 단계에서 무료배송 또는 할인 쿠폰을 노출한다.',
    'cart_to_checkout',
    'banner',
    '{"content_type": "coupon_banner", "cta_label": "쿠폰 받기"}'::jsonb
),
(
    'checkout_coupon_banner',
    '결제 직전 전환 쿠폰',
    '결제 직전 이탈이 높은 세그먼트에게 제한 시간 쿠폰 또는 마감 임박 메시지를 노출한다.',
    'checkout_to_purchase',
    'banner',
    '{"content_type": "coupon_banner", "cta_label": "지금 구매하기"}'::jsonb
),
(
    'alternative_product_banner',
    '대체 상품 추천 배너',
    '품절 또는 관심 상품 이탈이 높은 세그먼트에게 대체 상품을 추천한다.',
    'view_to_cart',
    'banner',
    '{"content_type": "product_recommendation_banner", "cta_label": "대체 상품 보기"}'::jsonb
)
ON CONFLICT (action_key) DO UPDATE
SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    target_funnel_step = EXCLUDED.target_funnel_step,
    default_channel = EXCLUDED.default_channel,
    template_json = EXCLUDED.template_json,
    is_active = true;

-- =========================================================
-- 3. Dummy decision run
-- This run acts as the source run for sample rows below.
-- =========================================================

INSERT INTO decision_runs (
    project_id,
    run_type,
    trigger_source,
    requested_by,
    idempotency_key,
    mode,
    force,
    analysis_date,
    window_start,
    window_end,
    baseline_start,
    baseline_end,
    status,
    metadata,
    started_at,
    finished_at
)
SELECT
    p.id,
    'manual_api',
    'api',
    'dummy-seed',
    'dummy-2021-01-04',
    'demo',
    true,
    DATE '2021-01-04',
    TIMESTAMPTZ '2021-01-04 00:00:00+09',
    TIMESTAMPTZ '2021-01-05 00:00:00+09',
    TIMESTAMPTZ '2021-01-01 00:00:00+09',
    TIMESTAMPTZ '2021-01-04 00:00:00+09',
    'success',
    '{"source": "dummy.sql", "purpose": "dashboard_and_ad_server_development"}'::jsonb,
    now() - interval '5 minutes',
    now() - interval '4 minutes'
FROM projects p
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, idempotency_key)
WHERE idempotency_key IS NOT NULL
DO UPDATE
SET
    status = 'success',
    metadata = EXCLUDED.metadata,
    finished_at = now();

-- =========================================================
-- 4. Default segment
-- =========================================================

INSERT INTO segments (
    project_id,
    segment_key,
    name,
    description,
    rule_json,
    status,
    is_default,
    created_run_id
)
SELECT
    p.id,
    'default',
    '전체 사용자 기본 세그먼트',
    '세그먼트별 추천 광고가 없을 때 사용하는 기본 fallback 세그먼트',
    '{"type": "default", "matches": "all"}'::jsonb,
    'active',
    true,
    r.id
FROM projects p
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_key) DO UPDATE
SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    rule_json = EXCLUDED.rule_json,
    status = 'active',
    is_default = true,
    updated_at = now();

-- =========================================================
-- 5. Default generated content
-- =========================================================

INSERT INTO generated_contents (
    project_id,
    segment_id,
    recommendation_action_id,
    content_type,
    variant_key,
    title,
    body,
    cta_label,
    landing_url,
    image_url,
    media_s3_key,
    image_prompt,
    generation_model,
    generation_status,
    metadata,
    created_run_id
)
SELECT
    p.id,
    s.id,
    NULL,
    'banner',
    'default',
    '오늘의 인기 상품을 확인해보세요',
    '고객님을 위한 추천 상품과 특별 혜택을 준비했습니다.',
    '추천 상품 보기',
    '/collections/recommended',
    '/static/banners/default-main-banner.png',
    NULL,
    'clean ecommerce promotional banner for popular products',
    'seed',
    'approved',
    '{"source": "dummy.sql", "purpose": "default_fallback"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'default'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, variant_key)
WHERE recommendation_action_id IS NULL
DO UPDATE
SET
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    cta_label = EXCLUDED.cta_label,
    landing_url = EXCLUDED.landing_url,
    image_url = EXCLUDED.image_url,
    image_prompt = EXCLUDED.image_prompt,
    generation_model = 'seed',
    generation_status = 'approved',
    metadata = EXCLUDED.metadata,
    updated_at = now();

-- =========================================================
-- 6. Default ad mapping
-- =========================================================

INSERT INTO segment_ad_mappings (
    project_id,
    segment_id,
    placement_key,
    experiment_id,
    experiment_variant_id,
    generated_content_id,
    traffic_weight,
    is_active,
    is_winner,
    priority,
    created_run_id
)
SELECT
    p.id,
    s.id,
    'main_banner',
    NULL,
    NULL,
    c.id,
    1.0,
    true,
    true,
    0,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'default'
JOIN generated_contents c
  ON c.project_id = p.id
 AND c.segment_id = s.id
 AND c.variant_key = 'default'
 AND c.recommendation_action_id IS NULL
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, placement_key)
WHERE experiment_variant_id IS NULL
DO UPDATE
SET
    generated_content_id = EXCLUDED.generated_content_id,
    traffic_weight = 1.0,
    is_active = true,
    is_winner = true,
    priority = 0,
    updated_at = now();

-- =========================================================
-- 7. Sample AI-defined segments
-- One normal segment and one anomalous segment.
-- =========================================================

INSERT INTO segments (
    project_id,
    segment_key,
    name,
    description,
    rule_json,
    status,
    is_default,
    created_run_id
)
SELECT
    p.id,
    'age_30s__gender_male__channel_kakao__category_fresh',
    '카카오톡 유입 / 30대 / 남성 / 신선식품 관심 사용자',
    '상품 조회 대비 장바구니 전환이 낮아 AI 개선 액션 대상이 된 세그먼트',
    '{"age_group":"30s","gender":"male","acquisition_channel":"kakao","primary_category":"fresh"}'::jsonb,
    'active',
    false,
    r.id
FROM projects p
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_key) DO UPDATE
SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    rule_json = EXCLUDED.rule_json,
    status = 'active',
    is_default = false,
    updated_at = now();

INSERT INTO segments (
    project_id,
    segment_key,
    name,
    description,
    rule_json,
    status,
    is_default,
    created_run_id
)
SELECT
    p.id,
    'age_20s__gender_female__channel_instagram__category_beauty',
    '인스타그램 유입 / 20대 / 여성 / 뷰티 관심 사용자',
    '이상 징후 없이 지표만 저장되는 정상 세그먼트 예시',
    '{"age_group":"20s","gender":"female","acquisition_channel":"instagram","primary_category":"beauty"}'::jsonb,
    'active',
    false,
    r.id
FROM projects p
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_key) DO UPDATE
SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    rule_json = EXCLUDED.rule_json,
    status = 'active',
    is_default = false,
    updated_at = now();

-- =========================================================
-- 8. Sample users and memberships
-- =========================================================

INSERT INTO user_profiles (
    project_id,
    external_user_id,
    gender,
    age_group,
    device_type,
    acquisition_channel,
    last_seen_at,
    properties
)
SELECT p.id, v.external_user_id, v.gender, v.age_group, v.device_type, v.acquisition_channel, v.last_seen_at, v.properties
FROM projects p
CROSS JOIN (
    VALUES
    ('user_kakao_30_male_001', 'male', '30s', 'mobile', 'kakao', TIMESTAMPTZ '2021-01-04 11:20:00+09', '{"primary_category":"fresh"}'::jsonb),
    ('user_kakao_30_male_002', 'male', '30s', 'mobile', 'kakao', TIMESTAMPTZ '2021-01-04 12:10:00+09', '{"primary_category":"fresh"}'::jsonb),
    ('user_instagram_20_female_001', 'female', '20s', 'mobile', 'instagram', TIMESTAMPTZ '2021-01-04 19:30:00+09', '{"primary_category":"beauty"}'::jsonb)
) AS v(external_user_id, gender, age_group, device_type, acquisition_channel, last_seen_at, properties)
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, external_user_id) DO UPDATE
SET
    gender = EXCLUDED.gender,
    age_group = EXCLUDED.age_group,
    device_type = EXCLUDED.device_type,
    acquisition_channel = EXCLUDED.acquisition_channel,
    last_seen_at = EXCLUDED.last_seen_at,
    properties = EXCLUDED.properties,
    updated_at = now();

INSERT INTO user_segment_memberships (
    project_id,
    external_user_id,
    segment_id,
    analysis_date,
    is_primary,
    confidence,
    reason_json,
    created_run_id
)
SELECT
    p.id,
    u.external_user_id,
    s.id,
    DATE '2021-01-04',
    true,
    1.0,
    jsonb_build_object('source', 'dummy.sql', 'matched_segment_key', s.segment_key),
    r.id
FROM projects p
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
JOIN user_profiles u
  ON u.project_id = p.id
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = CASE
    WHEN u.external_user_id LIKE 'user_kakao_30_male_%'
      THEN 'age_30s__gender_male__channel_kakao__category_fresh'
    ELSE 'age_20s__gender_female__channel_instagram__category_beauty'
 END
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, external_user_id, segment_id, analysis_date) DO UPDATE
SET
    is_primary = true,
    confidence = EXCLUDED.confidence,
    reason_json = EXCLUDED.reason_json;

-- =========================================================
-- 9. Segment daily metrics
-- Normal segment has no anomaly.
-- Anomalous segment has view_to_purchase_rate below target 5%.
-- =========================================================

INSERT INTO segment_daily_metrics (
    project_id,
    segment_id,
    analysis_date,
    user_count,
    session_count,
    page_view_count,
    product_view_count,
    add_to_cart_count,
    checkout_start_count,
    purchase_count,
    ad_impression_count,
    ad_click_count,
    revenue,
    view_to_cart_rate,
    cart_to_checkout_rate,
    checkout_to_purchase_rate,
    view_to_purchase_rate,
    ctr,
    cvr,
    baseline_view_to_purchase_rate,
    target_view_to_purchase_rate,
    metric_json,
    created_run_id
)
SELECT
    p.id,
    s.id,
    DATE '2021-01-04',
    430,
    610,
    1850,
    1000,
    90,
    50,
    25,
    820,
    66,
    1250000.00,
    0.090000,
    0.555556,
    0.500000,
    0.025000,
    0.080488,
    0.030488,
    0.060000,
    0.050000,
    '{"note":"anomalous sample: low view_to_cart and low final conversion"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, analysis_date) DO UPDATE
SET
    user_count = EXCLUDED.user_count,
    session_count = EXCLUDED.session_count,
    page_view_count = EXCLUDED.page_view_count,
    product_view_count = EXCLUDED.product_view_count,
    add_to_cart_count = EXCLUDED.add_to_cart_count,
    checkout_start_count = EXCLUDED.checkout_start_count,
    purchase_count = EXCLUDED.purchase_count,
    ad_impression_count = EXCLUDED.ad_impression_count,
    ad_click_count = EXCLUDED.ad_click_count,
    revenue = EXCLUDED.revenue,
    view_to_cart_rate = EXCLUDED.view_to_cart_rate,
    cart_to_checkout_rate = EXCLUDED.cart_to_checkout_rate,
    checkout_to_purchase_rate = EXCLUDED.checkout_to_purchase_rate,
    view_to_purchase_rate = EXCLUDED.view_to_purchase_rate,
    ctr = EXCLUDED.ctr,
    cvr = EXCLUDED.cvr,
    baseline_view_to_purchase_rate = EXCLUDED.baseline_view_to_purchase_rate,
    target_view_to_purchase_rate = EXCLUDED.target_view_to_purchase_rate,
    metric_json = EXCLUDED.metric_json;

INSERT INTO segment_daily_metrics (
    project_id,
    segment_id,
    analysis_date,
    user_count,
    session_count,
    page_view_count,
    product_view_count,
    add_to_cart_count,
    checkout_start_count,
    purchase_count,
    ad_impression_count,
    ad_click_count,
    revenue,
    view_to_cart_rate,
    cart_to_checkout_rate,
    checkout_to_purchase_rate,
    view_to_purchase_rate,
    ctr,
    cvr,
    baseline_view_to_purchase_rate,
    target_view_to_purchase_rate,
    metric_json,
    created_run_id
)
SELECT
    p.id,
    s.id,
    DATE '2021-01-04',
    320,
    460,
    1320,
    700,
    165,
    98,
    48,
    610,
    71,
    2110000.00,
    0.235714,
    0.593939,
    0.489796,
    0.068571,
    0.116393,
    0.078689,
    0.061000,
    0.050000,
    '{"note":"normal sample: no anomaly should be generated"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_20s__gender_female__channel_instagram__category_beauty'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, analysis_date) DO UPDATE
SET
    user_count = EXCLUDED.user_count,
    session_count = EXCLUDED.session_count,
    page_view_count = EXCLUDED.page_view_count,
    product_view_count = EXCLUDED.product_view_count,
    add_to_cart_count = EXCLUDED.add_to_cart_count,
    checkout_start_count = EXCLUDED.checkout_start_count,
    purchase_count = EXCLUDED.purchase_count,
    ad_impression_count = EXCLUDED.ad_impression_count,
    ad_click_count = EXCLUDED.ad_click_count,
    revenue = EXCLUDED.revenue,
    view_to_cart_rate = EXCLUDED.view_to_cart_rate,
    cart_to_checkout_rate = EXCLUDED.cart_to_checkout_rate,
    checkout_to_purchase_rate = EXCLUDED.checkout_to_purchase_rate,
    view_to_purchase_rate = EXCLUDED.view_to_purchase_rate,
    ctr = EXCLUDED.ctr,
    cvr = EXCLUDED.cvr,
    baseline_view_to_purchase_rate = EXCLUDED.baseline_view_to_purchase_rate,
    target_view_to_purchase_rate = EXCLUDED.target_view_to_purchase_rate,
    metric_json = EXCLUDED.metric_json;

-- =========================================================
-- 10. Anomaly and root cause for anomalous segment only
-- =========================================================

INSERT INTO segment_anomalies (
    project_id,
    segment_id,
    analysis_date,
    metric_name,
    actual_value,
    expected_value,
    target_value,
    difference_value,
    difference_rate,
    severity,
    impact_score,
    status,
    evidence_json,
    created_run_id
)
SELECT
    p.id,
    s.id,
    DATE '2021-01-04',
    'view_to_purchase_rate',
    0.025000,
    0.060000,
    0.050000,
    -0.025000,
    -0.500000,
    'high',
    0.875000,
    'detected',
    '{"dominant_drop":"view_to_cart","product_view_count":1000,"add_to_cart_count":90,"hypothesis":"benefit visibility or stockout issue"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, analysis_date, metric_name) DO UPDATE
SET
    actual_value = EXCLUDED.actual_value,
    expected_value = EXCLUDED.expected_value,
    target_value = EXCLUDED.target_value,
    difference_value = EXCLUDED.difference_value,
    difference_rate = EXCLUDED.difference_rate,
    severity = EXCLUDED.severity,
    impact_score = EXCLUDED.impact_score,
    status = 'detected',
    evidence_json = EXCLUDED.evidence_json;

INSERT INTO root_cause_candidates (
    anomaly_id,
    cause_type,
    cause_key,
    title,
    description,
    confidence_score,
    impact_score,
    rank_no,
    evidence_json
)
SELECT
    a.id,
    'funnel_step_drop',
    'view_to_cart',
    '상품 조회 후 장바구니 추가 전환율 저하',
    '상품을 조회한 사용자 대비 장바구니에 추가한 사용자가 적어 최종 구매 전환율이 목표보다 낮습니다.',
    0.8200,
    0.875000,
    1,
    '{"view_to_cart_rate":0.09,"expected_view_to_cart_rate":0.18,"recommended_action_key":"highlight_benefit_banner"}'::jsonb
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN segment_anomalies a
  ON a.project_id = p.id
 AND a.segment_id = s.id
 AND a.analysis_date = DATE '2021-01-04'
 AND a.metric_name = 'view_to_purchase_rate'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (anomaly_id, cause_type, cause_key) DO UPDATE
SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    confidence_score = EXCLUDED.confidence_score,
    impact_score = EXCLUDED.impact_score,
    rank_no = EXCLUDED.rank_no,
    evidence_json = EXCLUDED.evidence_json;

-- =========================================================
-- 11. Recommendation result and action
-- =========================================================

INSERT INTO recommendation_results (
    project_id,
    segment_id,
    anomaly_id,
    primary_root_cause_id,
    analysis_date,
    summary,
    status,
    recommendation_json,
    created_run_id
)
SELECT
    p.id,
    s.id,
    a.id,
    rc.id,
    DATE '2021-01-04',
    '카카오톡 유입 30대 남성 신선식품 세그먼트에서 상품 조회 후 장바구니 추가 전환율이 낮습니다. 혜택 강조 배너 실험을 추천합니다.',
    'experiment_running',
    '{"selected_action_keys":["highlight_benefit_banner"],"reason":"low view_to_cart_rate"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN segment_anomalies a
  ON a.project_id = p.id
 AND a.segment_id = s.id
 AND a.analysis_date = DATE '2021-01-04'
 AND a.metric_name = 'view_to_purchase_rate'
JOIN root_cause_candidates rc
  ON rc.anomaly_id = a.id
 AND rc.cause_key = 'view_to_cart'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, analysis_date, anomaly_id)
WHERE anomaly_id IS NOT NULL
DO UPDATE
SET
    primary_root_cause_id = EXCLUDED.primary_root_cause_id,
    summary = EXCLUDED.summary,
    status = EXCLUDED.status,
    recommendation_json = EXCLUDED.recommendation_json,
    updated_at = now();

INSERT INTO recommendation_actions (
    recommendation_result_id,
    project_id,
    segment_id,
    action_catalog_id,
    action_key,
    title,
    description,
    priority,
    expected_effect_metric,
    expected_effect_direction,
    expected_effect_value,
    status,
    metadata
)
SELECT
    rr.id,
    p.id,
    s.id,
    ac.id,
    'highlight_benefit_banner',
    '신선식품 혜택 강조 배너 노출',
    '상품 리스트와 메인 배너에서 무료배송/특가 혜택을 강조해 장바구니 추가를 유도합니다.',
    1,
    'view_to_purchase_rate',
    'increase',
    0.015000,
    'running',
    '{"source":"dummy.sql","target_funnel_step":"view_to_cart"}'::jsonb
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN segment_anomalies a
  ON a.project_id = p.id
 AND a.segment_id = s.id
 AND a.analysis_date = DATE '2021-01-04'
 AND a.metric_name = 'view_to_purchase_rate'
JOIN recommendation_results rr
  ON rr.project_id = p.id
 AND rr.segment_id = s.id
 AND rr.analysis_date = DATE '2021-01-04'
 AND rr.anomaly_id = a.id
JOIN action_catalog ac
  ON ac.action_key = 'highlight_benefit_banner'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (recommendation_result_id, action_key) DO UPDATE
SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    priority = EXCLUDED.priority,
    expected_effect_metric = EXCLUDED.expected_effect_metric,
    expected_effect_direction = EXCLUDED.expected_effect_direction,
    expected_effect_value = EXCLUDED.expected_effect_value,
    status = EXCLUDED.status,
    metadata = EXCLUDED.metadata,
    updated_at = now();

-- =========================================================
-- 12. Generated contents for control and treatment
-- =========================================================

INSERT INTO generated_contents (
    project_id,
    segment_id,
    recommendation_action_id,
    content_type,
    variant_key,
    title,
    body,
    cta_label,
    landing_url,
    image_url,
    image_prompt,
    generation_model,
    generation_status,
    metadata,
    created_run_id
)
SELECT
    p.id,
    s.id,
    ra.id,
    'banner',
    'control',
    '신선식품 인기 상품 모음',
    '오늘 많이 찾는 신선식품을 한눈에 확인해보세요.',
    '상품 보기',
    '/collections/fresh',
    '/static/banners/fresh-control.png',
    'simple fresh food ecommerce banner',
    'dummy',
    'approved',
    '{"source":"dummy.sql","role":"control"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, recommendation_action_id, variant_key)
WHERE recommendation_action_id IS NOT NULL
DO UPDATE
SET
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    cta_label = EXCLUDED.cta_label,
    landing_url = EXCLUDED.landing_url,
    image_url = EXCLUDED.image_url,
    image_prompt = EXCLUDED.image_prompt,
    generation_model = EXCLUDED.generation_model,
    generation_status = EXCLUDED.generation_status,
    metadata = EXCLUDED.metadata,
    updated_at = now();

INSERT INTO generated_contents (
    project_id,
    segment_id,
    recommendation_action_id,
    content_type,
    variant_key,
    title,
    body,
    cta_label,
    landing_url,
    image_url,
    image_prompt,
    generation_model,
    generation_status,
    metadata,
    created_run_id
)
SELECT
    p.id,
    s.id,
    ra.id,
    'banner',
    'treatment_a',
    '오늘 신선식품 무료배송 혜택을 확인하세요',
    '장바구니에 담기 전, 카카오톡 유입 고객 전용 신선식품 특가와 무료배송 혜택을 확인해보세요.',
    '혜택 상품 보기',
    '/collections/fresh?promo=free-shipping',
    '/static/banners/fresh-benefit-treatment-a.png',
    'fresh food promotional banner, free shipping badge, clean ecommerce style',
    'dummy',
    'approved',
    '{"source":"dummy.sql","role":"treatment","action_key":"highlight_benefit_banner"}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, recommendation_action_id, variant_key)
WHERE recommendation_action_id IS NOT NULL
DO UPDATE
SET
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    cta_label = EXCLUDED.cta_label,
    landing_url = EXCLUDED.landing_url,
    image_url = EXCLUDED.image_url,
    image_prompt = EXCLUDED.image_prompt,
    generation_model = EXCLUDED.generation_model,
    generation_status = EXCLUDED.generation_status,
    metadata = EXCLUDED.metadata,
    updated_at = now();

-- =========================================================
-- 13. Experiment and variants
-- =========================================================

INSERT INTO experiments (
    project_id,
    segment_id,
    recommendation_action_id,
    name,
    objective_metric,
    target_value,
    allocation_policy,
    status,
    start_date,
    end_date,
    decision_rule_json,
    created_run_id
)
SELECT
    p.id,
    s.id,
    ra.id,
    '신선식품 혜택 강조 배너 실험',
    'click_to_purchase_rate',
    0.050000,
    'fixed_split',
    'running',
    DATE '2021-01-04',
    NULL,
    '{"winner_rule":"target_value_and_min_sample","min_impressions":100,"min_conversions":10}'::jsonb,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, recommendation_action_id)
WHERE recommendation_action_id IS NOT NULL
DO UPDATE
SET
    name = EXCLUDED.name,
    objective_metric = EXCLUDED.objective_metric,
    target_value = EXCLUDED.target_value,
    allocation_policy = EXCLUDED.allocation_policy,
    status = 'running',
    decision_rule_json = EXCLUDED.decision_rule_json,
    updated_at = now();

INSERT INTO experiment_variants (
    experiment_id,
    project_id,
    variant_key,
    name,
    generated_content_id,
    is_control,
    traffic_weight,
    impression_count,
    click_count,
    conversion_count,
    ctr,
    conversion_rate,
    status
)
SELECT
    e.id,
    p.id,
    'control',
    '기존 신선식품 배너',
    c.id,
    true,
    0.5000,
    410,
    28,
    1,
    0.068293,
    0.035714,
    'active'
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN experiments e
  ON e.project_id = p.id
 AND e.recommendation_action_id = ra.id
JOIN generated_contents c
  ON c.project_id = p.id
 AND c.recommendation_action_id = ra.id
 AND c.variant_key = 'control'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (experiment_id, variant_key) DO UPDATE
SET
    name = EXCLUDED.name,
    generated_content_id = EXCLUDED.generated_content_id,
    is_control = EXCLUDED.is_control,
    traffic_weight = EXCLUDED.traffic_weight,
    impression_count = EXCLUDED.impression_count,
    click_count = EXCLUDED.click_count,
    conversion_count = EXCLUDED.conversion_count,
    ctr = EXCLUDED.ctr,
    conversion_rate = EXCLUDED.conversion_rate,
    status = EXCLUDED.status,
    updated_at = now();

INSERT INTO experiment_variants (
    experiment_id,
    project_id,
    variant_key,
    name,
    generated_content_id,
    is_control,
    traffic_weight,
    impression_count,
    click_count,
    conversion_count,
    ctr,
    conversion_rate,
    status
)
SELECT
    e.id,
    p.id,
    'treatment_a',
    '혜택 강조 배너',
    c.id,
    false,
    0.5000,
    410,
    38,
    1,
    0.092683,
    0.026316,
    'active'
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN experiments e
  ON e.project_id = p.id
 AND e.recommendation_action_id = ra.id
JOIN generated_contents c
  ON c.project_id = p.id
 AND c.recommendation_action_id = ra.id
 AND c.variant_key = 'treatment_a'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (experiment_id, variant_key) DO UPDATE
SET
    name = EXCLUDED.name,
    generated_content_id = EXCLUDED.generated_content_id,
    is_control = EXCLUDED.is_control,
    traffic_weight = EXCLUDED.traffic_weight,
    impression_count = EXCLUDED.impression_count,
    click_count = EXCLUDED.click_count,
    conversion_count = EXCLUDED.conversion_count,
    ctr = EXCLUDED.ctr,
    conversion_rate = EXCLUDED.conversion_rate,
    status = EXCLUDED.status,
    updated_at = now();

-- =========================================================
-- 14. Segment ad mappings for experiment variants
-- Advertisement server reads these rows and uses traffic_weight for split.
-- =========================================================

INSERT INTO segment_ad_mappings (
    project_id,
    segment_id,
    placement_key,
    experiment_id,
    experiment_variant_id,
    generated_content_id,
    traffic_weight,
    is_active,
    is_winner,
    priority,
    created_run_id
)
SELECT
    p.id,
    s.id,
    'main_banner',
    e.id,
    ev.id,
    ev.generated_content_id,
    ev.traffic_weight,
    true,
    false,
    100,
    r.id
FROM projects p
JOIN segments s
  ON s.project_id = p.id
 AND s.segment_key = 'age_30s__gender_male__channel_kakao__category_fresh'
JOIN recommendation_actions ra
  ON ra.project_id = p.id
 AND ra.segment_id = s.id
 AND ra.action_key = 'highlight_benefit_banner'
JOIN experiments e
  ON e.project_id = p.id
 AND e.recommendation_action_id = ra.id
JOIN experiment_variants ev
  ON ev.experiment_id = e.id
JOIN decision_runs r
  ON r.project_id = p.id
 AND r.idempotency_key = 'dummy-2021-01-04'
WHERE p.project_key = 'demo-shop'
ON CONFLICT (project_id, segment_id, placement_key, experiment_variant_id)
DO UPDATE
SET
    experiment_id = EXCLUDED.experiment_id,
    generated_content_id = EXCLUDED.generated_content_id,
    traffic_weight = EXCLUDED.traffic_weight,
    is_active = true,
    is_winner = false,
    priority = 100,
    updated_at = now();
