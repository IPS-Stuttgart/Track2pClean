# Coherence teacher rescue follow-up candidates

The current strongest Track2pPolicy-family row is the coherence suffix stitch followed by Track2p-teacher adjacent rescue.  To probe whether residual adjacent-FN or missing-seed repairs can improve it, run the checked-in next-steps manifest and compare the existing coherence suffix teacher row against dynamic-confidence, target-extension, and seed-source variants exposed by `track2p-policy-coherence-suffix-teacher-rescue`.

`coherence_teacher_action_specific.json` is a focused probe for the next safest
teacher-assisted improvement attempt. It keeps the coherence suffix stitch row
fixed, then uses dynamic seed/cell confidence with separate gates for target
extensions and seed-source backfills. This lets the residual-FN and missing-seed
hypotheses be tested without admitting the broad teacher-edge tail.
