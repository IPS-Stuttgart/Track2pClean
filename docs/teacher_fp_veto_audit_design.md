# Teacher FP veto audit design

ComponentCleanup leaves a small pairwise false-positive budget. The next diagnostic should inspect residual edges that are present after ComponentCleanup but absent from Track2p, simulate a local split/veto at each edge, and report official pairwise and complete-track deltas before any prediction-changing teacher-veto method is added.
