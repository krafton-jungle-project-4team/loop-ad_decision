-- postgres-init.sql
-- 테이블 생성 전용. seed 데이터는 넣지 않는다.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =========================================================
-- 1. Projects
-- =========================================================

CREATE TABLE IF NOT EXISTS projects (
    id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    sdk_key VARCHAR(255) UNIQUE NOT NULL DEFAULT encode(gen_random_bytes(24), 'hex'),
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_projects_updated_at ON projects;
CREATE TRIGGER trg_projects_updated_at
BEFORE UPDATE ON projects
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 2. Dashboard Users
-- =========================================================

CREATE TABLE IF NOT EXISTS dashboard_users (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    password_hash TEXT,
    role VARCHAR(64) NOT NULL DEFAULT 'admin',
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, email)
);

CREATE INDEX IF NOT EXISTS idx_dashboard_users_project
ON dashboard_users (project_id);

DROP TRIGGER IF EXISTS trg_dashboard_users_updated_at ON dashboard_users;
CREATE TRIGGER trg_dashboard_users_updated_at
BEFORE UPDATE ON dashboard_users
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 3. User Profiles
-- =========================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    external_user_id VARCHAR(255) NOT NULL,

    age_group VARCHAR(32),
    gender VARCHAR(32),
    membership_level VARCHAR(64),

    attributes_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, external_user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_project_user
ON user_profiles (project_id, external_user_id);

DROP TRIGGER IF EXISTS trg_user_profiles_updated_at ON user_profiles;
CREATE TRIGGER trg_user_profiles_updated_at
BEFORE UPDATE ON user_profiles
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 4. Segments
-- =========================================================

CREATE TABLE IF NOT EXISTS segments (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    conditions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    segment_hash VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, segment_hash)
);

CREATE INDEX IF NOT EXISTS idx_segments_project_status
ON segments (project_id, status);

CREATE INDEX IF NOT EXISTS gin_segments_conditions_json
ON segments USING GIN (conditions_json);

DROP TRIGGER IF EXISTS trg_segments_updated_at ON segments;
CREATE TRIGGER trg_segments_updated_at
BEFORE UPDATE ON segments
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 5. Campaigns
-- =========================================================

CREATE TABLE IF NOT EXISTS campaigns (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    external_campaign_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    channel VARCHAR(64),
    goal VARCHAR(64),
    budget NUMERIC(18, 2),
    status VARCHAR(32) NOT NULL DEFAULT 'active',

    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, external_campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_campaigns_project_status
ON campaigns (project_id, status);

CREATE INDEX IF NOT EXISTS idx_campaigns_project_channel
ON campaigns (project_id, channel);

DROP TRIGGER IF EXISTS trg_campaigns_updated_at ON campaigns;
CREATE TRIGGER trg_campaigns_updated_at
BEFORE UPDATE ON campaigns
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 6. Coupons
-- =========================================================

CREATE TABLE IF NOT EXISTS coupons (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    code VARCHAR(255),
    name VARCHAR(255) NOT NULL,

    discount_type VARCHAR(64) NOT NULL,
    discount_rate NUMERIC(5, 4),
    discount_amount NUMERIC(18, 2),
    max_discount_amount NUMERIC(18, 2),

    budget NUMERIC(18, 2),
    status VARCHAR(32) NOT NULL DEFAULT 'active',

    starts_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, code)
);

CREATE INDEX IF NOT EXISTS idx_coupons_project_status
ON coupons (project_id, status);

DROP TRIGGER IF EXISTS trg_coupons_updated_at ON coupons;
CREATE TRIGGER trg_coupons_updated_at
BEFORE UPDATE ON coupons
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 7. Ad Creatives
-- =========================================================

CREATE TABLE IF NOT EXISTS ad_creatives (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    campaign_id BIGINT REFERENCES campaigns(id) ON DELETE SET NULL,
    coupon_id BIGINT REFERENCES coupons(id) ON DELETE SET NULL,

    action_id VARCHAR(128),
    creative_type VARCHAR(64) NOT NULL DEFAULT 'banner',

    title VARCHAR(255),
    message TEXT,
    image_url TEXT,
    landing_url TEXT,

    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    status VARCHAR(32) NOT NULL DEFAULT 'active',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ad_creatives_project_status
ON ad_creatives (project_id, status);

CREATE INDEX IF NOT EXISTS idx_ad_creatives_project_action_status
ON ad_creatives (project_id, action_id, status);

CREATE INDEX IF NOT EXISTS idx_ad_creatives_campaign
ON ad_creatives (campaign_id);

DROP TRIGGER IF EXISTS trg_ad_creatives_updated_at ON ad_creatives;
CREATE TRIGGER trg_ad_creatives_updated_at
BEFORE UPDATE ON ad_creatives
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 8. Action Catalog
-- =========================================================

CREATE TABLE IF NOT EXISTS action_catalog (
    action_id VARCHAR(128) PRIMARY KEY,
    action_type VARCHAR(64) NOT NULL,

    title VARCHAR(255) NOT NULL,
    description TEXT,
    target_step VARCHAR(128),

    base_weight DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    primary_metric VARCHAR(128),
    expected_impact TEXT,

    execution_hint_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    status VARCHAR(32) NOT NULL DEFAULT 'active',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_action_catalog_type_status
ON action_catalog (action_type, status);

DROP TRIGGER IF EXISTS trg_action_catalog_updated_at ON action_catalog;
CREATE TRIGGER trg_action_catalog_updated_at
BEFORE UPDATE ON action_catalog
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 9. Automation Policies
-- =========================================================

CREATE TABLE IF NOT EXISTS automation_policies (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,

    enabled BOOLEAN NOT NULL DEFAULT false,
    auto_execute_enabled BOOLEAN NOT NULL DEFAULT false,

    allowed_action_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    allowed_action_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    blocked_action_ids JSONB NOT NULL DEFAULT '[]'::jsonb,

    max_experiment_traffic_ratio DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    min_priority_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    max_discount_rate DOUBLE PRECISION,
    max_daily_coupon_budget DOUBLE PRECISION,
    max_message_per_user_per_day BIGINT,
    stop_loss_relative_drop DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_automation_policies_project
ON automation_policies (project_id);

DROP TRIGGER IF EXISTS trg_automation_policies_updated_at ON automation_policies;
CREATE TRIGGER trg_automation_policies_updated_at
BEFORE UPDATE ON automation_policies
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 10. Recommendation Results
-- =========================================================

CREATE TABLE IF NOT EXISTS recommendation_results (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    baseline_start TIMESTAMPTZ,
    baseline_end TIMESTAMPTZ,

    segment_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    segment_hash VARCHAR(64) NOT NULL,

    status VARCHAR(64) NOT NULL,

    anomaly_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    root_causes_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    recommendations_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_decision_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    decision_by VARCHAR(255),
    decision_at TIMESTAMPTZ,
    decision_reason TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_project_status
ON recommendation_results (project_id, status);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_project_created
ON recommendation_results (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_segment_hash
ON recommendation_results (segment_hash);

CREATE INDEX IF NOT EXISTS gin_recommendation_results_segment_json
ON recommendation_results USING GIN (segment_json);

DROP TRIGGER IF EXISTS trg_recommendation_results_updated_at ON recommendation_results;
CREATE TRIGGER trg_recommendation_results_updated_at
BEFORE UPDATE ON recommendation_results
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 11. Analysis Jobs
-- 비동기 통합 분석 요청과 worker 처리 상태
-- =========================================================

CREATE TABLE IF NOT EXISTS analysis_jobs (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    request_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    recommendation_result_id BIGINT
        REFERENCES recommendation_results(id) ON DELETE SET NULL,

    error_message TEXT,
    attempts BIGINT NOT NULL DEFAULT 0,
    max_attempts BIGINT NOT NULL DEFAULT 1,

    locked_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_project_id
ON analysis_jobs (project_id);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status
ON analysis_jobs (status);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_recommendation_result
ON analysis_jobs (recommendation_result_id);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status_created
ON analysis_jobs (status, created_at);

DROP TRIGGER IF EXISTS trg_analysis_jobs_updated_at ON analysis_jobs;
CREATE TRIGGER trg_analysis_jobs_updated_at
BEFORE UPDATE ON analysis_jobs
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 12. Experiments
-- =========================================================

CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    recommendation_result_id BIGINT NOT NULL
        REFERENCES recommendation_results(id) ON DELETE CASCADE,

    segment_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    segment_hash VARCHAR(64) NOT NULL,

    action_id VARCHAR(128) NOT NULL,
    action_type VARCHAR(64) NOT NULL,

    status VARCHAR(64) NOT NULL,

    traffic_split_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    primary_metric VARCHAR(128),
    guardrail_metrics_json JSONB NOT NULL DEFAULT '[]'::jsonb,

    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_experiments_recommendation_action
        UNIQUE (recommendation_result_id, action_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_project_status
ON experiments (project_id, status);

CREATE INDEX IF NOT EXISTS idx_experiments_recommendation
ON experiments (recommendation_result_id);

CREATE INDEX IF NOT EXISTS idx_experiments_segment_hash
ON experiments (segment_hash);

DROP TRIGGER IF EXISTS trg_experiments_updated_at ON experiments;
CREATE TRIGGER trg_experiments_updated_at
BEFORE UPDATE ON experiments
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- 13. Segment Ad Mappings
-- 추천 서버가 쓰고, 광고 서버가 직접 읽는 핵심 테이블
-- =========================================================

CREATE TABLE IF NOT EXISTS segment_ad_mappings (
    id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    segment_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    segment_hash VARCHAR(64) NOT NULL,

    recommendation_result_id BIGINT NOT NULL
        REFERENCES recommendation_results(id) ON DELETE CASCADE,

    experiment_id BIGINT
        REFERENCES experiments(id) ON DELETE SET NULL,

    campaign_id BIGINT
        REFERENCES campaigns(id) ON DELETE SET NULL,

    creative_id BIGINT
        REFERENCES ad_creatives(id) ON DELETE SET NULL,

    coupon_id BIGINT
        REFERENCES coupons(id) ON DELETE SET NULL,

    action_id VARCHAR(128) NOT NULL,
    action_type VARCHAR(64) NOT NULL,

    execution_hint_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    status VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL,

    expires_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_segment_ad_mappings_recommendation_action
        UNIQUE (recommendation_result_id, action_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_project_status
ON segment_ad_mappings (project_id, status);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_project_segment_status
ON segment_ad_mappings (project_id, segment_hash, status);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_recommendation
ON segment_ad_mappings (recommendation_result_id);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_experiment
ON segment_ad_mappings (experiment_id);

CREATE INDEX IF NOT EXISTS gin_segment_ad_mappings_segment_json
ON segment_ad_mappings USING GIN (segment_json);

DROP TRIGGER IF EXISTS trg_segment_ad_mappings_updated_at ON segment_ad_mappings;
CREATE TRIGGER trg_segment_ad_mappings_updated_at
BEFORE UPDATE ON segment_ad_mappings
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
