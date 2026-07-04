# Loop-Ad Decision API

Loop-Ad Decision API is a lifecycle write API service for analysis, generation,
promotion run creation, segment assignment, evaluation, and next-loop
orchestration.

The service is not a Dashboard API, ChatKit API, or advertisement-serving
Decision hot path. Dashboard-owned systems handle segment query preview,
ChatKit flows, banner resolve, redirect handling, dispatch, public read APIs,
and any public recommendation-style API surface.

## Serving Boundary

Dashboard and ad execution must not synchronously call Decision for per-request
serving. They should read the contract database directly. When available, they
should read the Data Source Contract owned `active_ad_serving_assignments` view.

Decision does not provide active_ad_serving_assignments and does not own that
view. The data-source contract owns database schemas and serving views.

## Next Loop Integration Note

B6 next-loop currently defines the decision-side orchestration and the
analysis/generation call boundary. The real analysis and generation adapters are
left for a follow-up integration PR after the analysis and generation flows are
ready to honor failed segment focus inputs end to end.
