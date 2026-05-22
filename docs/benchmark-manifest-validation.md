# Benchmark Manifest Validation

Use `bayescatrack benchmark validate-suite` to parse a benchmark manifest, resolve run and comparison outputs, and print a plan without executing the benchmark.

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json
```

For machine-readable output:

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json --format json
```

By default, configured `data` and `reference` paths do not need to exist. This lets ordinary CI validate manifests intended for dedicated benchmark runners. To require paths to exist on the current machine:

```bash
python -m bayescatrack benchmark validate-suite benchmarks.json --check-input-paths
```

The validator accepts the same `--output-dir` and `--progress` overrides as `bayescatrack benchmark suite`, so both commands resolve the manifest consistently.
