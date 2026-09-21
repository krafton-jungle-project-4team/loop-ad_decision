# Local artifact boundary

Large raw measurements, database snapshots, charts, PDFs, and temporary
rendered files are intentionally not committed.

The canonical local source for this checkpoint is:

```text
artifacts/ann-search/scale-series-v2/ann-efficiency-recovery/
```

The committed summary records each required source file's relative path,
byte length, and SHA-256 digest. Run the evidence validator with
`--verify-local-artifacts` to compare the local files to those digests.

Do not commit:

- raw JSONL measurements;
- database or cohort snapshots;
- generated charts, PDFs, slide assets, or previews;
- temporary checkpoints that are not part of the final decision;
- machine-specific connection information.

If raw artifacts need durable team storage later, upload an immutable archive
to team-controlled object storage and add its URI and digest to the summary.
