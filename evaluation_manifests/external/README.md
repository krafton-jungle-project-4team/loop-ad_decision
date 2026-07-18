# External sealed manifests

Commit each external final-evaluation manifest as
`<manifest_id>.json` in this directory. The manifest is the public, auditable
evaluation contract. Each evaluator downloads the official source files
independently. The files and frozen model are accepted only when their size and
SHA-256 match this manifest.

Do not commit source datasets, outcome files, model binaries, execution journals,
or result artifacts here.
