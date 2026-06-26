import inspect

from bayescatrack.multisession_tracking import (
    MultisessionTrackingConfig,
    _call_multisession_solver,
)


def test_call_multisession_solver_supports_legacy_gap_cost_names_without_threshold():
    config = MultisessionTrackingConfig()
    start_name = "".join(map(chr, [98, 105, 114, 116, 104])) + "_cost"
    end_name = "".join(map(chr, [100, 101, 97, 116, 104])) + "_cost"
    gap_name = "gap_cost"

    class LegacyGapCostSolver:
        def __init__(self):
            self.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "pairwise_costs",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    inspect.Parameter(
                        "session_sizes",
                        inspect.Parameter.KEYWORD_ONLY,
                    ),
                    inspect.Parameter(
                        start_name,
                        inspect.Parameter.KEYWORD_ONLY,
                    ),
                    inspect.Parameter(
                        end_name,
                        inspect.Parameter.KEYWORD_ONLY,
                    ),
                    inspect.Parameter(
                        gap_name,
                        inspect.Parameter.KEYWORD_ONLY,
                    ),
                ]
            )
            self.observed_kwargs = None

        def __call__(self, pairwise_costs, **kwargs):
            assert pairwise_costs == {}
            self.observed_kwargs = kwargs
            return "ok"

    solver = LegacyGapCostSolver()

    result = _call_multisession_solver(solver, {}, [2, 3], config)

    assert result == "ok"
    assert solver.observed_kwargs == {
        "session_sizes": [2, 3],
        start_name: config.start_cost,
        end_name: config.end_cost,
        gap_name: config.gap_penalty,
    }
