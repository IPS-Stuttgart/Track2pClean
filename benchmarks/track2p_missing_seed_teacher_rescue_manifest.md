# Missing-seed teacher rescue candidate

This focused benchmark candidate targets residual complete-track false negatives caused by missing seed-session source observations after Track2pPolicy component cleanup.

It should be run with the existing `track2p-policy-teacher-adjacent-rescue` command using `--teacher-repair-preset missing-seed-high-confidence`, `--no-allow-source-backfill`, and `--allow-fragment-merges`.
