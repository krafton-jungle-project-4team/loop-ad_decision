CREATE TABLE IF NOT EXISTS events
(
    project_id String,
    event_id String,
    user_id Nullable(String),
    session_id String,
    event_time DateTime64(3, 'Asia/Seoul'),
    event_name LowCardinality(String),

    channel Nullable(String),
    campaign_id Nullable(String),
    age_group Nullable(String),
    gender Nullable(String),
    device Nullable(String),
    category Nullable(String),
    product_id Nullable(String),
    inventory_status Nullable(String),

    price Nullable(Float64),
    quantity Nullable(Int64),
    revenue Nullable(Float64),
    coupon_id Nullable(String),
    order_id Nullable(String)
)
ENGINE = MergeTree
ORDER BY (project_id, event_time, session_id, event_name);
