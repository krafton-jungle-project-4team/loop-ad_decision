# Content Generation PR Breakdown

## PR 1. Domain Service and Unit Rules

Goal: implement the content generation core without database or daily-job coupling.

Includes:
- `ContentGenerationService.generate_for_actions(...)`
- deterministic `MockContentGenerator`
- optional `OpenAIContentGenerator` behind env/config
- prompt sanitization that excludes raw events and user-level identifiers
- force/default/status rules
- repository protocol and no-op lock boundary
- unit tests with an in-memory fake repository

Does not include:
- PostgreSQL SQL implementation
- daily job orchestration
- public or internal HTTP API
- new experiment or mapping creation

Acceptance:
- recommended non-default actions generate `control` and `treatment_a`
- default segment and seed/default content are protected
- `force=false` skips existing content
- `force=true` can regenerate AI-created content
- failures mark `recommendation_actions.status = failed`

## PR 2. PostgreSQL Repository

Goal: connect the PR 1 service to the real AI Decision schema.

Includes:
- target query for anomalous, non-default recommendation actions
- `generated_contents` upsert by `recommendation_action_id + variant_key`
- `generated_contents.created_run_id` persistence from `run_id`
- advisory lock or `SELECT ... FOR UPDATE`
- action status and metadata updates

Does not include:
- creating new experiments or mappings
- linking generated content into experiment variants or serving mappings
- HTTP endpoint

Acceptance:
- rerunning the same action is idempotent
- failed action metadata stores `content_generation_failed`
- seed/default content rows are never updated

## PR 3. Daily Job Integration

Goal: call content generation from the daily decision flow after recommendation actions exist.

Includes:
- invoke `ContentGenerationService` after recommendation action creation
- pass `project_id`, `analysis_date`, `run_id`, and `force`
- continue the job when individual content generation fails
- expose summary in job metadata/logging

Does not include:
- content debug API
- creating experiment/mapping business logic inside the content service

Acceptance:
- anomaly-free runs create no new content
- anomaly runs create content before experiment/mapping finalization
- failed content actions do not fail the whole daily job

## Deferred. Experiment and Mapping Linking

Goal: keep experiment and serving mapping creation in the recommendation/experiment ownership boundary.

Content generation does not create or update:
- `experiments`
- `experiment_variants`
- `segment_ad_mappings`

Those responsibilities should live in the recommendation/experiment flow, where schema constraints,
traffic policy, and serving contracts are already owned.

## PR 5. Optional Internal Debug Trigger

Goal: add a debug-only internal trigger if the team needs one.

Includes:
- `POST /internal/contents/generate`
- `X-Admin-Token` validation against `AI_DECISION_ADMIN_TOKEN`
- calls the same service path as the daily job

Does not include:
- public recommendation or ad-serving API
- dashboard or advertisement server dependency on AI Decision
