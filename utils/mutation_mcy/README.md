# MCY Mutation Runner Modules

This directory holds reusable components for `run_mutation_mcy_examples.sh`.

## Stable Runner API
Use `utils/run_mutation_mcy_examples_api.sh` as the stable automation entrypoint.

Profiles:
- `default`: pass-through behavior.
- `native-real`: native mutation backend + real harness mode (strict, no smoke/yosys).

## Native Components
- `lib/native_mutation_plan.py`: generates deterministic native mutation label plans.
- `templates/native_create_mutated.py`: native mutation rewrite engine template copied per example run.

These modules are intended to reduce inline bash complexity while preserving
existing runner behavior.
