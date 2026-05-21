# Benchmark Manifest Validation

Use `bayescatrack benchmark validate-suite` to parse a benchmark manifest, resolve all run and comparison outputs, and print the execution plan without running Track2p scoring or registration.

This is useful for CI checks on expensive benchmark suites because it catches malformed keys, incompatible runner options, duplicate names, invalid output formats, and unresolved manifest structure before a data-dependent run starts.

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json
```

The command prints a Markdown plan by default. For machine-readable CI output, use JSON:

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json --format json
```

By default, missing `data` and `reference` paths are tolerated. This allows manifests for hosted runners, mounted datasets, or lab-local paths to be validated in ordinary CI. To additionally require paths to exist on the current machine, add:

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json --check-input-paths
```

The validator accepts the same `--output-dir` and `--progress` overrides as `bayescatrack benchmark suite`, so both commands resolve the manifest consistently.
