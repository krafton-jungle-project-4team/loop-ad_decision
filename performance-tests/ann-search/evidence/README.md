# Evidence format

`experiment-index.json` is the entry point. Each experiment links to:

- `summary.json`: machine-readable scope, gates, results, limitations,
  provenance, and source hashes;
- `report.md`: a short technical interpretation of the same result.

An experiment may be preserved even when it fails or needs follow-up work.
Its status and claim boundary must remain explicit. This checkpoint uses
`follow_up_required` because candidate retrieval passed while corrected
full-membership evaluation did not produce a winner.
