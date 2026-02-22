# Whole-Project Ownership Map

Last updated: February 20, 2026

This map defines ownership boundaries for long-term maintainability of runner,
formal, simulation, and mutation infrastructure.

## Ownership Model

- `Primary owner`: domain that approves behavior changes.
- `Secondary owner`: domain that reviews integration impact.
- `Shared contract`: files/interfaces that require cross-domain review.

Owner groups below are capability-based (not individual names) so the map stays
stable across agent/personnel changes.

## Domain Map

| Domain | Primary Owner | Secondary Owner | Key Paths |
|---|---|---|---|
| Runner Core | `runner-infra` | `formal`, `mutation`, `simulation` | `utils/lib/`, `utils/refactor_continue.sh` |
| Mutation Flows | `mutation` | `runner-infra` | `utils/run_mutation_*`, `utils/mutation_mcy/`, `docs/MutationRunnerContract.md` |
| Formal BMC/LEC | `formal` | `runner-infra` | `utils/run_*_circt_bmc*`, `utils/run_*_circt_lec*`, `docs/FormalRunnerContract.md` |
| Simulation Wrappers | `simulation` | `runner-infra` | `utils/run_avip_*`, `utils/run_sv_tests_*sim*`, `docs/SimulationRunnerContract.md` |
| Unified Orchestrator | `runner-infra` | `formal`, `simulation`, `mutation` | `utils/run_regression_unified.sh`, `docs/UNIFIED_REGRESSION_ORCHESTRATOR.md`, `docs/unified_regression_*.tsv` |
| `circt-sim` Runtime | `sim-runtime` | `simulation` | `tools/circt-sim/`, `include/circt/Dialect/Sim/`, `lib/Dialect/Sim/` |
| Schema/Artifacts | `schema-governance` | all runner domains | summary/retry/parity TSV headers + schema sidecars |
| Tool Tests/Fixtures | `test-infra` | corresponding domain | `test/Tools/`, `test/Tools/Inputs/` |

## Shared Contracts Requiring Cross-Domain Review

1. `docs/MutationRunnerContract.md`
2. `docs/FormalRunnerContract.md`
3. `docs/SimulationRunnerContract.md`
4. `docs/UNIFIED_REGRESSION_ORCHESTRATOR.md`
5. `docs/unified_regression_manifest.tsv`
6. `docs/unified_regression_adapter_catalog.tsv`

## Change Routing Rules

1. `utils/lib/` changes: require `runner-infra` + at least one affected domain review.
2. Schema/header changes in runner outputs: require `schema-governance` and update corresponding contract docs/tests.
3. `run_regression_unified.sh` CLI changes: require `runner-infra` and orchestrator lit updates.
4. `tools/circt-sim/` behavior changes: require `sim-runtime` and focused lit/unit validation.

## Merge-Conflict Minimization Rules

1. Prefer adding new helper modules over editing monolithic scripts directly.
2. Keep adapter catalog/manifest changes in dedicated commits.
3. For large refactors, split contract updates from behavior updates.
4. When touching shared files, include a concise rationale in changelog entry.

## Escalation Matrix

- Failing mutation-only gates: `mutation` leads.
- Failing BMC/LEC-only gates: `formal` leads.
- Failing AVIP/sim wrapper gates: `simulation` leads.
- Cross-domain schema or orchestrator failures: `runner-infra` leads with `schema-governance` co-review.
