# MCY Mutation Runner Modules

This directory holds reusable components for `run_mutation_mcy_examples.sh`.

## Stable Runner API
Use `utils/run_mutation_mcy_examples_api.sh` as the stable automation entrypoint.

Profiles:
- `default`: pass-through behavior.
- `native-real`: native mutation backend + real harness mode (strict, no smoke/yosys).

## Native Components
- `lib/native_mutation_plan.sh`: delegates native mutation planning to
  `circt-mut generate` (C++ planner implementation).
- native mutation apply now uses `circt-mut apply` directly via a tiny shell
  wrapper generated per example run.

These modules are intended to reduce inline bash complexity while preserving
existing runner behavior.
