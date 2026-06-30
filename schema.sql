-- =========================================================
-- Loop-Ad AI Decision PostgreSQL Schema
-- =========================================================
--
-- Role of this database:
--   PostgreSQL is the contract DB read by Dashboard and Advertisement servers.
--   AI Decision writes analysis/recommendation/content/experiment results here.
--
-- Important architecture rule:
--   Dashboard and Advertisement servers must NOT call AI Decision for serving.
--   They only read PostgreSQL.
--
-- Job model:
--   Production cron/EventBridge runs once per day.
--   For demo/manual verification, an internal admin API triggers the same job path.
--   Manual API trigger is not a public recommendation/serving API.
--
-- Data source:
--   ClickHouse remains the raw event source.
--   Do not copy raw ClickHouse events into PostgreSQL.
--
-- Fallback rule:
--   Default segment + default content + default mapping must always exist.
--   Users without segment membership must still receive the default ad.
--
-- =========================================================

-- =========================================================
-- 0. Projects
-- =========================================================

CREATE TABLE IF NOT EXISTS projects (
    id BIGSERIAL PRIMARY KEY,
    project_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Seoul',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================================================
-- 1. Decision Runs
-- One row per cron/manual/API execution.
-- The manual API creates a row with run_type = manual_api.
-- =========================================================

CREATE TABLE IF NOT EXISTS decision_runs (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    run_type TEXT NOT NULL DEFAULT 'daily_cron'
        CHECK (run_type IN ('daily_cron', 'manual_api', 'manual_cli')),

    trigger_source TEXT NOT NULL DEFAULT 'cron'
        CHECK (trigger_source IN ('cron', 'api', 'cli', 'test')),

    requested_by TEXT,
    idempotency_key TEXT,

    mode TEXT NOT NULL DEFAULT 'normal'
        CHECK (mode IN ('normal', 'demo', 'backfill')),

    force BOOLEAN NOT NULL DEFAULT false,

    analysis_date DATE NOT NULL,

    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    baseline_start TIMESTAMPTZ,
    baseline_end TIMESTAMPTZ,

    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('queued', 'running', 'success', 'failed', 'skipped')),

    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_decision_runs_project_date
ON decision_runs (project_id, analysis_date DESC, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_runs_status
ON decision_runs (project_id, status, started_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_decision_runs_idempotency_key
ON decision_runs (project_id, idempotency_key)
WHERE idempotency_key IS NOT NULL;

-- =========================================================
-- 2. User Profiles
-- Minimal user metadata. This is not a raw event copy table.
-- =========================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    external_user_id TEXT NOT NULL,

    gender TEXT,
    age_group TEXT,
    device_type TEXT,
    acquisition_channel TEXT,

    last_seen_at TIMESTAMPTZ,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, external_user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_project_user
ON user_profiles (project_id, external_user_id);

-- =========================================================
-- 3. Segments
-- AI-defined segment definitions.
-- A project must have exactly one default segment for ad fallback.
-- =========================================================

CREATE TABLE IF NOT EXISTS segments (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    segment_key TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,

    rule_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive')),

    is_default BOOLEAN NOT NULL DEFAULT false,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, segment_key)
);

CREATE INDEX IF NOT EXISTS idx_segments_project_status
ON segments (project_id, status);

CREATE UNIQUE INDEX IF NOT EXISTS uq_segments_one_default_per_project
ON segments (project_id)
WHERE is_default = true;

-- =========================================================
-- 4. User Segment Memberships
-- Advertisement server reads this to find a user's primary segment.
-- If no row exists, Advertisement server must fallback to default segment.
-- =========================================================

CREATE TABLE IF NOT EXISTS user_segment_memberships (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    external_user_id TEXT NOT NULL,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    analysis_date DATE NOT NULL,

    is_primary BOOLEAN NOT NULL DEFAULT true,
    confidence NUMERIC(5,4) NOT NULL DEFAULT 1.0000,

    reason_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, external_user_id, segment_id, analysis_date)
);

CREATE INDEX IF NOT EXISTS idx_user_segment_lookup
ON user_segment_memberships (project_id, external_user_id, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_user_segment_segment
ON user_segment_memberships (project_id, segment_id, analysis_date DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_one_primary_segment_per_user_per_day
ON user_segment_memberships (project_id, external_user_id, analysis_date)
WHERE is_primary = true;

-- =========================================================
-- 5. Segment Daily Metrics
-- Saved for every valid segment every day, even when there is no anomaly.
-- Dashboard reads this table for charts.
-- =========================================================

CREATE TABLE IF NOT EXISTS segment_daily_metrics (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    analysis_date DATE NOT NULL,

    user_count INTEGER NOT NULL DEFAULT 0,
    session_count INTEGER NOT NULL DEFAULT 0,

    page_view_count INTEGER NOT NULL DEFAULT 0,
    product_view_count INTEGER NOT NULL DEFAULT 0,
    add_to_cart_count INTEGER NOT NULL DEFAULT 0,
    checkout_start_count INTEGER NOT NULL DEFAULT 0,
    purchase_count INTEGER NOT NULL DEFAULT 0,

    ad_impression_count INTEGER NOT NULL DEFAULT 0,
    ad_click_count INTEGER NOT NULL DEFAULT 0,

    revenue NUMERIC(14,2) NOT NULL DEFAULT 0,

    view_to_cart_rate NUMERIC(12,6),
    cart_to_checkout_rate NUMERIC(12,6),
    checkout_to_purchase_rate NUMERIC(12,6),
    view_to_purchase_rate NUMERIC(12,6),

    ctr NUMERIC(12,6),
    cvr NUMERIC(12,6),

    baseline_view_to_purchase_rate NUMERIC(12,6),
    target_view_to_purchase_rate NUMERIC(12,6) NOT NULL DEFAULT 0.050000,

    metric_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, segment_id, analysis_date)
);

CREATE INDEX IF NOT EXISTS idx_segment_metrics_project_date
ON segment_daily_metrics (project_id, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_segment_metrics_segment_date
ON segment_daily_metrics (project_id, segment_id, analysis_date DESC);

-- =========================================================
-- 6. Segment Anomalies
-- Only anomalous segments receive rows here.
-- No anomaly means no row.
-- =========================================================

CREATE TABLE IF NOT EXISTS segment_anomalies (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    analysis_date DATE NOT NULL,

    metric_name TEXT NOT NULL,
    actual_value NUMERIC(12,6),
    expected_value NUMERIC(12,6),
    target_value NUMERIC(12,6),

    difference_value NUMERIC(12,6),
    difference_rate NUMERIC(12,6),

    severity TEXT NOT NULL DEFAULT 'low'
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),

    impact_score NUMERIC(12,6) NOT NULL DEFAULT 0,

    status TEXT NOT NULL DEFAULT 'detected'
        CHECK (status IN ('detected', 'resolved', 'ignored')),

    evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, segment_id, analysis_date, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_segment_anomalies_project_date
ON segment_anomalies (project_id, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_segment_anomalies_segment
ON segment_anomalies (project_id, segment_id, status);

-- =========================================================
-- 7. Root Cause Candidates
-- Causes are created only for anomalies.
-- =========================================================

CREATE TABLE IF NOT EXISTS root_cause_candidates (
    id BIGSERIAL PRIMARY KEY,
    anomaly_id BIGINT NOT NULL REFERENCES segment_anomalies(id) ON DELETE CASCADE,

    cause_type TEXT NOT NULL,
    cause_key TEXT NOT NULL,

    title TEXT NOT NULL,
    description TEXT,

    confidence_score NUMERIC(5,4) NOT NULL DEFAULT 0.5000,
    impact_score NUMERIC(12,6) NOT NULL DEFAULT 0,
    rank_no INTEGER NOT NULL DEFAULT 1,

    evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (anomaly_id, cause_type, cause_key)
);

CREATE INDEX IF NOT EXISTS idx_root_causes_anomaly_rank
ON root_cause_candidates (anomaly_id, rank_no);

-- =========================================================
-- 8. Action Catalog
-- Seed-managed action menu.
-- recommendation_actions are the actions actually selected for a segment.
-- =========================================================

CREATE TABLE IF NOT EXISTS action_catalog (
    id BIGSERIAL PRIMARY KEY,

    action_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,

    target_funnel_step TEXT,
    default_channel TEXT NOT NULL DEFAULT 'banner',

    template_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_action_catalog_active
ON action_catalog (is_active, action_key);

-- =========================================================
-- 9. Recommendation Results
-- Parent summary of recommendation reasoning for one anomaly/segment.
-- Think of this as the prescription summary.
-- =========================================================

CREATE TABLE IF NOT EXISTS recommendation_results (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    anomaly_id BIGINT REFERENCES segment_anomalies(id) ON DELETE SET NULL,
    primary_root_cause_id BIGINT REFERENCES root_cause_candidates(id) ON DELETE SET NULL,

    analysis_date DATE NOT NULL,

    summary TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending_content'
        CHECK (
            status IN (
                'no_action',
                'pending_content',
                'content_generated',
                'experiment_ready',
                'experiment_running',
                'winner_selected',
                'dismissed'
            )
        ),

    recommendation_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_project_date
ON recommendation_results (project_id, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_segment
ON recommendation_results (project_id, segment_id, analysis_date DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendation_result_per_anomaly
ON recommendation_results (project_id, segment_id, analysis_date, anomaly_id)
WHERE anomaly_id IS NOT NULL;

-- =========================================================
-- 10. Recommendation Actions
-- Child executable actions under recommendation_results.
-- One result can have multiple actions.
-- =========================================================

CREATE TABLE IF NOT EXISTS recommendation_actions (
    id BIGSERIAL PRIMARY KEY,
    recommendation_result_id BIGINT NOT NULL REFERENCES recommendation_results(id) ON DELETE CASCADE,

    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    action_catalog_id BIGINT REFERENCES action_catalog(id) ON DELETE SET NULL,

    action_key TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,

    priority INTEGER NOT NULL DEFAULT 1,

    expected_effect_metric TEXT NOT NULL DEFAULT 'view_to_purchase_rate',
    expected_effect_direction TEXT NOT NULL DEFAULT 'increase'
        CHECK (expected_effect_direction IN ('increase', 'decrease')),
    expected_effect_value NUMERIC(12,6),

    status TEXT NOT NULL DEFAULT 'recommended'
        CHECK (
            status IN (
                'recommended',
                'content_generated',
                'experiment_created',
                'running',
                'won',
                'lost',
                'dismissed',
                'failed'
            )
        ),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (recommendation_result_id, action_key)
);

CREATE INDEX IF NOT EXISTS idx_recommendation_actions_status
ON recommendation_actions (project_id, status);

CREATE INDEX IF NOT EXISTS idx_recommendation_actions_segment
ON recommendation_actions (project_id, segment_id, status);

-- =========================================================
-- 11. Generated Contents
-- Renderable content for ads. Includes seed default content and generated content.
-- =========================================================

CREATE TABLE IF NOT EXISTS generated_contents (
    id BIGSERIAL PRIMARY KEY,

    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    recommendation_action_id BIGINT REFERENCES recommendation_actions(id) ON DELETE SET NULL,

    content_type TEXT NOT NULL DEFAULT 'banner'
        CHECK (
            content_type IN (
                'banner',
                'coupon_banner',
                'product_recommendation_banner',
                'push_message',
                'email'
            )
        ),

    variant_key TEXT NOT NULL,

    title TEXT NOT NULL,
    body TEXT,
    cta_label TEXT,
    landing_url TEXT,

    image_url TEXT,
    media_s3_key TEXT,
    image_prompt TEXT,

    generation_model TEXT,
    generation_status TEXT NOT NULL DEFAULT 'generated'
        CHECK (generation_status IN ('generated', 'failed', 'approved')),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generated_contents_action
ON generated_contents (recommendation_action_id);

CREATE INDEX IF NOT EXISTS idx_generated_contents_segment
ON generated_contents (project_id, segment_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_generated_content_per_action_variant
ON generated_contents (project_id, recommendation_action_id, variant_key)
WHERE recommendation_action_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_default_generated_content_per_segment_variant
ON generated_contents (project_id, segment_id, variant_key)
WHERE recommendation_action_id IS NULL;

-- =========================================================
-- 12. Experiments
-- Created for recommendation actions on anomalous segments.
-- Existing running experiments are updated every daily/manual run.
-- =========================================================

CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,

    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    recommendation_action_id BIGINT REFERENCES recommendation_actions(id) ON DELETE SET NULL,

    name TEXT NOT NULL,

    objective_metric TEXT NOT NULL DEFAULT 'click_to_purchase_rate',
    target_value NUMERIC(12,6) NOT NULL DEFAULT 0.050000,

    allocation_policy TEXT NOT NULL DEFAULT 'fixed_split'
        CHECK (
            allocation_policy IN (
                'fixed_split',
                'thompson_sampling',
                'winner_take_all'
            )
        ),

    status TEXT NOT NULL DEFAULT 'running'
        CHECK (
            status IN (
                'draft',
                'running',
                'paused',
                'completed',
                'winner_selected'
            )
        ),

    start_date DATE NOT NULL,
    end_date DATE,

    winner_variant_id BIGINT,

    decision_rule_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiments_project_status
ON experiments (project_id, status);

CREATE INDEX IF NOT EXISTS idx_experiments_segment
ON experiments (project_id, segment_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_per_recommendation_action
ON experiments (project_id, recommendation_action_id)
WHERE recommendation_action_id IS NOT NULL;

-- =========================================================
-- 13. Experiment Variants
-- control/treatment rows. traffic_weight is used by Advertisement server.
-- alpha/beta are kept for later Thompson Sampling extension.
-- =========================================================

CREATE TABLE IF NOT EXISTS experiment_variants (
    id BIGSERIAL PRIMARY KEY,

    experiment_id BIGINT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    variant_key TEXT NOT NULL,
    name TEXT NOT NULL,

    generated_content_id BIGINT REFERENCES generated_contents(id) ON DELETE SET NULL,

    is_control BOOLEAN NOT NULL DEFAULT false,

    traffic_weight NUMERIC(5,4) NOT NULL DEFAULT 0.5000
        CHECK (traffic_weight >= 0 AND traffic_weight <= 1),

    alpha NUMERIC(12,6) NOT NULL DEFAULT 1.000000,
    beta NUMERIC(12,6) NOT NULL DEFAULT 1.000000,

    impression_count INTEGER NOT NULL DEFAULT 0,
    click_count INTEGER NOT NULL DEFAULT 0,
    conversion_count INTEGER NOT NULL DEFAULT 0,

    ctr NUMERIC(12,6),
    conversion_rate NUMERIC(12,6),

    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'winner', 'loser')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (experiment_id, variant_key)
);

CREATE INDEX IF NOT EXISTS idx_experiment_variants_experiment
ON experiment_variants (experiment_id);

CREATE INDEX IF NOT EXISTS idx_experiment_variants_content
ON experiment_variants (generated_content_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_experiments_winner_variant'
    ) THEN
        ALTER TABLE experiments
        ADD CONSTRAINT fk_experiments_winner_variant
        FOREIGN KEY (winner_variant_id)
        REFERENCES experiment_variants(id)
        ON DELETE SET NULL;
    END IF;
END $$;

-- =========================================================
-- 14. Segment Ad Mappings
-- Final serving contract. Advertisement server reads this through active_ad_serving_rules.
-- Segment-specific mappings have priority over the default segment fallback.
-- =========================================================

CREATE TABLE IF NOT EXISTS segment_ad_mappings (
    id BIGSERIAL PRIMARY KEY,

    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,

    placement_key TEXT NOT NULL DEFAULT 'main_banner',

    experiment_id BIGINT REFERENCES experiments(id) ON DELETE SET NULL,
    experiment_variant_id BIGINT REFERENCES experiment_variants(id) ON DELETE SET NULL,
    generated_content_id BIGINT REFERENCES generated_contents(id) ON DELETE SET NULL,

    traffic_weight NUMERIC(5,4) NOT NULL DEFAULT 1.0000
        CHECK (traffic_weight >= 0 AND traffic_weight <= 1),

    is_active BOOLEAN NOT NULL DEFAULT true,
    is_winner BOOLEAN NOT NULL DEFAULT false,

    priority INTEGER NOT NULL DEFAULT 100,

    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until TIMESTAMPTZ,

    created_run_id BIGINT REFERENCES decision_runs(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, segment_id, placement_key, experiment_variant_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_active
ON segment_ad_mappings (project_id, segment_id, placement_key, is_active, priority DESC);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_experiment
ON segment_ad_mappings (experiment_id, experiment_variant_id);

CREATE INDEX IF NOT EXISTS idx_segment_ad_mappings_content
ON segment_ad_mappings (generated_content_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_default_segment_ad_mapping
ON segment_ad_mappings (project_id, segment_id, placement_key)
WHERE experiment_variant_id IS NULL;

-- =========================================================
-- 15. Views
-- =========================================================

CREATE OR REPLACE VIEW latest_user_primary_segments AS
SELECT DISTINCT ON (m.project_id, m.external_user_id)
    m.project_id,
    m.external_user_id,
    m.segment_id,
    s.segment_key,
    s.name AS segment_name,
    m.analysis_date,
    m.confidence,
    m.reason_json,
    m.created_at
FROM user_segment_memberships m
JOIN segments s
  ON s.id = m.segment_id
WHERE m.is_primary = true
ORDER BY m.project_id, m.external_user_id, m.analysis_date DESC, m.created_at DESC;

CREATE OR REPLACE VIEW active_ad_serving_rules AS
SELECT
    m.project_id,
    p.project_key,
    m.id AS mapping_id,
    m.segment_id,
    s.segment_key,
    s.name AS segment_name,
    s.is_default AS is_default_segment,

    m.placement_key,

    m.experiment_id,
    m.experiment_variant_id,
    ev.variant_key,

    m.generated_content_id,
    c.content_type,
    c.title,
    c.body,
    c.cta_label,
    c.landing_url,
    c.image_url,
    c.media_s3_key,
    c.image_prompt,

    m.traffic_weight,
    m.is_winner,
    m.priority,
    m.valid_from,
    m.valid_until
FROM segment_ad_mappings m
JOIN projects p
    ON p.id = m.project_id
JOIN segments s
    ON s.id = m.segment_id
LEFT JOIN experiment_variants ev
    ON ev.id = m.experiment_variant_id
LEFT JOIN generated_contents c
    ON c.id = m.generated_content_id
WHERE m.is_active = true
  AND now() >= m.valid_from
  AND (m.valid_until IS NULL OR now() < m.valid_until);
