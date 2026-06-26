-- clickhouse-init.sql
-- 테이블 생성 전용. seed 이벤트는 넣지 않는다.

-- =========================================================
-- 1. Raw Events
-- =========================================================

CREATE TABLE IF NOT EXISTS events
(
    project_id LowCardinality(String),

    event_id String,
    user_id String,
    session_id String,

    event_time DateTime64(3, 'Asia/Seoul'),

    event_name LowCardinality(String),

    channel LowCardinality(String) DEFAULT '',
    campaign_id String DEFAULT '',

    age_group LowCardinality(String) DEFAULT '',
    gender LowCardinality(String) DEFAULT '',
    device LowCardinality(String) DEFAULT '',

    category String DEFAULT '',
    product_id String DEFAULT '',
    inventory_status LowCardinality(String) DEFAULT '',

    price Decimal(18, 2) DEFAULT 0,
    quantity UInt32 DEFAULT 0,
    revenue Decimal(18, 2) DEFAULT 0,

    coupon_id String DEFAULT '',
    order_id String DEFAULT '',

    experiment_id String DEFAULT '',
    variant_id LowCardinality(String) DEFAULT '',
    action_id String DEFAULT '',
    mapping_id String DEFAULT '',
    ad_id String DEFAULT '',
    creative_id String DEFAULT '',

    properties_json String DEFAULT '',

    ingested_at DateTime64(3, 'Asia/Seoul') DEFAULT now64(3, 'Asia/Seoul'),

    INDEX idx_event_id event_id TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX idx_user_id user_id TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX idx_session_id session_id TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX idx_product_id product_id TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX idx_experiment_id experiment_id TYPE bloom_filter(0.01) GRANULARITY 4
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (
    project_id,
    event_time,
    event_name,
    session_id,
    user_id,
    product_id
);

-- =========================================================
-- 2. 5분 단위 이벤트 카운트 집계
-- =========================================================

CREATE TABLE IF NOT EXISTS event_counts_5m
(
    bucket_start DateTime64(3, 'Asia/Seoul'),

    project_id LowCardinality(String),
    event_name LowCardinality(String),

    channel LowCardinality(String),
    campaign_id String,

    device LowCardinality(String),
    category String,
    product_id String,
    inventory_status LowCardinality(String),

    event_count UInt64,
    revenue_sum Decimal(18, 2)
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(bucket_start)
ORDER BY (
    project_id,
    bucket_start,
    event_name,
    channel,
    campaign_id,
    device,
    category,
    product_id,
    inventory_status
);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_event_counts_5m
TO event_counts_5m
AS
SELECT
    toStartOfInterval(event_time, INTERVAL 5 MINUTE) AS bucket_start,

    project_id,
    event_name,

    channel,
    campaign_id,

    device,
    category,
    product_id,
    inventory_status,

    count() AS event_count,
    sum(revenue) AS revenue_sum
FROM events
GROUP BY
    bucket_start,
    project_id,
    event_name,
    channel,
    campaign_id,
    device,
    category,
    product_id,
    inventory_status;

-- =========================================================
-- 3. 실험 성과 일별 집계
-- =========================================================

CREATE TABLE IF NOT EXISTS experiment_metrics_daily
(
    event_date Date,

    project_id LowCardinality(String),

    experiment_id String,
    variant_id LowCardinality(String),
    action_id String,
    mapping_id String,
    creative_id String,

    impressions UInt64,
    clicks UInt64,
    purchases UInt64,
    coupon_uses UInt64,

    revenue_sum Decimal(18, 2)
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (
    project_id,
    event_date,
    experiment_id,
    variant_id,
    action_id,
    mapping_id,
    creative_id
);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_experiment_metrics_daily
TO experiment_metrics_daily
AS
SELECT
    toDate(event_time) AS event_date,

    project_id,

    experiment_id,
    variant_id,
    action_id,
    mapping_id,
    creative_id,

    countIf(event_name = 'ad_impression') AS impressions,
    countIf(event_name = 'ad_click') AS clicks,
    countIf(event_name = 'purchase') AS purchases,
    countIf(event_name = 'coupon_used') AS coupon_uses,

    sumIf(revenue, event_name = 'purchase') AS revenue_sum
FROM events
WHERE experiment_id != ''
GROUP BY
    event_date,
    project_id,
    experiment_id,
    variant_id,
    action_id,
    mapping_id,
    creative_id;
