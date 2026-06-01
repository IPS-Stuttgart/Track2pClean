# Teacher seed-FN rescue candidate

This note records the next accuracy candidate after the component residual audit: combine dynamic seed-source prioritization with the `track2p-fn-rescue` teacher feature preset.

Recommended row:

- runner: `track2p-policy-teacher-adjacent-rescue`
- `teacher_edge_order`: `dynamic-seed-confidence`
- `teacher_feature_preset`: `track2p-fn-rescue`
- `allow_source_backfill`: `true`
- `allow_seed_source_backfill`: `true`
- `allow_completing_seed_source_backfill`: `true`
- `allow_fragment_merges`: `true`
- `min_component_observations`: `2`
- `max_applied_edits`: `1` or `2`

The row targets the intersection of the two remaining residual-audit buckets: Track2p-supported adjacent false negatives and missing seed-session source observations.
