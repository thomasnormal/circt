# Upstream Cherry-Pick Log

This file tracks upstream `llvm/circt` mining so we can resume quickly without
re-triaging the same commits.

## Snapshot

- Date: 2026-02-28
- Local base used for mining: `origin/main` @ `9f24dc1d7`
- Upstream tip at time of scan: `upstream/main` @ `0f99432e5`
- Scope scanned: recent upstream-only commits in `origin/main..upstream/main`

## Picked (Current Staging Stack)

Branch: `staging-upstream-easy-picks`

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `c4e7ccef7` | `c1ef24514` | [Comb] Fix SextMatcher crash on block arguments |
| `e2145fea9` | `a0b58ccce` | [CI] Cancel in-progress PR builds on new push (#9751) |
| `bbfade290` | `49e7f3af0` | [HW] Make sure the index type for arrays is at least i1 (#9733) |
| `c20acf2c5` | `7a25c970c` | [HWToLLVM] Take the correct data layout alignment for alloca (#9734) |
| `a3122b983` | `1f510c056` | [Arc] Add time operations for LLHD simulation support (#9747) |
| `a6896edae` | `4cd8e5fe6` | [Arc] Lower llhd.current_time to Arc in LowerState (#9756) |
| `f542e3e5a` | `d395d44c7` | [Moore] Add ToBuiltinIntOp (#9533) |
| `993ec9360` | `11cc66244` | [Datapath] Bug Fix for Sign-Extension Logic when Lowering Partial Products to Booth Arrays (#9726) (#9744) |
| `760ddb91e` | `52b012089` | [ConvertToLLVM] Add hw::ConstantOp conversion support and tests (#9709) |
| `69b64e842` | `73e28e09a` | [CalyxToHW] Fix missing i1-to-clock conversion in convertPipelineOp (#9715) |
| `d1e5b472c` | `52c919970` | [FIRRTL] Allow full reset module instances outside of reset domain (#9754) |
| `dc8c1e05c` | `a0ddbccf0` | [FIRRTL] Fix domain info updates in cloneWithInsertedPorts (#9758) |
| `8c0c5a148` | `0f99432e5` | [Arc] Lower time operations to LLVM IR (#9757) |
| `7a7ee71ab` | n/a (follow-up) | [ArcToLLVM] Fix LLHD include path for time-op lowering |

## Triaged But Not Picked

### Already present (empty cherry-pick)

- `99eb20083` [Comb] Add a transform to ensure division does not trap/SIGFPE (#9619)
- `6ce4738d9` [Seq] Add clock_div(clock_div(clock, a), b) -> clock_div(clock, a+b) canonicalization (#9549)
- `3b30c823b` [HandshakeToHW] Fix crashes when multiple `sync`s with different numbers of `none` inputs are used (#9587)

### Conflicted in dry-run (needs manual integration)

- `afcefdc19` [comb] New Canonicalization ~sext(x) = sext(~x) (#9637)
- `db50d3131` [LLHD] Fix mem2reg to capture all live values across wait ops (#9552)

### Dropped in earlier iteration

- ImportVerilog picks (`f4e0bd1d3`, `08704d250`, `447b49e09`) dropped because
  this local baseline currently has Slang-front-end build constraints that
  prevented reliable validation in this staging environment.
- `6434ee4c1` ([HW][circt-reduce] sanitization) dropped because `circt-reduce`
  is not currently buildable in this staging setup due external MLIR archive
  mismatch.

## Validation Notes

- Built in staging tree (`build_stage`): `circt-opt`, `circt-translate`,
  `arcilator`.
- Previously validated in this branch:
  - Arc/Moore/HW/Comb/HWToLLVM focused tests via filtered `lit`.
- After each additional pick batch, rerun focused tests for changed files before
  promoting to `main`.

## Resume Pointer

For the next mining pass, start from these unresolved high-value candidates:

1. `db50d3131` ([LLHD] mem2reg live-value fix)
2. `afcefdc19` ([comb] ~sext canonicalization)

Command to refresh candidate window:

```bash
git log --oneline --no-merges origin/main..upstream/main
```

## Refresh (2026-02-28, later pass)

- Current staging head: `0a9d22958`
- Current upstream tip: `5f7d374a7`
- Stack size vs `origin/main`: `42` commits ahead

### Additional Picks Since The Initial Log

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `29f81dbaa` | `debd22694` | [FIRRTL] Add conservative IMDCE handling for InstanceChoiceOp (#9710) |
| `9010819e8` | `5ade31e47` | [FIRRTL] Support FInstanceLike operations in ModuleInliner (#9688) |
| `228e21581` | `70d66d7f4` | [FIRRTL][LowerToHW] Add InstanceChoiceOp lowering, Part 1 (#9742) |
| `0cbd42b69` | `274eeb55d` | [FIRRTL] Add domain create op (#9774) |
| `9c25e1e46` | `3d5455330` | [FIRRTL] Add instance macro attribute to InstanceChoice for Lowering (#9760) |
| `0f1439c64` | `15f3650af` | [FIRRTL] Change FInstanceLike to consider multiple referred modules (#9676) |
| `54045c245` | `ec285f538` | [firtool] Add --num-threads/-j option to control parallel compilation (#9551) |
| `7fac47214` | `e5d5eb6b0` | [SCFToCalyx] Fix incorrect assert in setResultRegs for scf::IfOp (#9721) |
| `4e5e4ff21` | `ee4badcde` | [circt-bmc] Add LTLToCore to pipeline (#9735) |
| `ecf557539` | `10fbfc9b1` | [FIRRTL][Reduce] Fix module-port-pruner crash with probe ports (#9694) |
| `7be632221` | `179c31994` | [circt-lec] Add lowering from Synth to Comb (#9725) |
| `8b6cb92f1` | `def39e7c7` | [FIRRTL] Add CheckCombLoops handling for InstanceChoiceOp (#9711) |
| `f9bfb6126` | `e3964f818` | [FIRRTL][IMCP] Add conservative support for InstanceChoiceOp (#9692) |
| `d7418d057` | `8f953acab` | [FIRRTL][LayerSink] Support InstanceChoice (#9696) |
| `ba1dbecf3` | `0e9f82038` | [FIRRTL][LowerLayers] Fix instance input port capture |
| `bdea0c823` | `b2ab3aff0` | [FIRRTL] Dedup: fix non-deduplicatable public module handling (#9702) |
| `116518665` | `e089efd1f` | [FIRRTL][SpecializeOption] Erase all options with default flag |
| `af6209414` | `9f9da0678` | [HWAggregateToComb] Support hw.sturct_extruct and hw.struct_create (#9675) |
| `9cf58aa41` | `36828e715` | Fix ambiguous call to ServiceImplRecordOp::create in ESIServices.cpp (#9707) |
| `8cd1ab33d` | `eb22f87b1` | [CombToSynth] Remove operation type restriction |
| `4e63160e3` | `f21dbe7c3` | [circt-reduce] Use per-port matching for FIRRTL port pruners (#9755) |
| `092be06f3` | `6542026a9` | [FIRRTL] Improve error messages for domain symbol verification (#9776) |
| `dffde85f9` | `6e3d168f6` | [ESI][Runtime] Don't crash on unsupported type (#9768) |
| `dc0cfd659` | `f348cf978` | [ESI][Runtime] Fix paths in Trace and XRT backends (#9740) |
| `2bb16b0e1` | `5f7d374a7` | [FIRRTL] Support merging layers in LinkCircuits (#9677) |
| `0a9d22958` | `6434ee4c1` | [HW][circt-reduce] Add HW name sanitization (#9730) |

Local follow-up commits kept in stack:

- `42eee6923` [Upstream] Fix follow-up conflicts in firtool and circt-lec picks

### Deferred In This Pass

- `17330f8a9` [Moore][ImportVerilog] fork-join lowering:
  conflicts with this branch's existing `moore.fork` model and would introduce
  a second incompatible fork op path.
- `d3ddbe121`, `79369384c`, `bc7cf6bfc` (ImportVerilog/Moore queue series):
  conflict-heavy on this baseline (multiple core files). Revisit in a dedicated
  ImportVerilog sync pass.

### Validation Added In This Pass

- Build:
  - `utils/ninja-with-lock.sh -C build_stage circt-opt circt-reduce`
- Lit tests:
  - `build_stage/test/Dialect/FIRRTL/errors.mlir`
  - `build_stage/test/Dialect/FIRRTL/link-layers.mlir`
  - `build_stage/test/Dialect/FIRRTL/link-layers-errors.mlir`
  - `build_stage/test/Dialect/FIRRTL/Reduction/pattern-registration.mlir`
  - `build_stage/test/Dialect/FIRRTL/Reduction/module-port-pruner.mlir`
  - `build_stage/test/Dialect/FIRRTL/Reduction/module-port-pruner-probe.mlir`
  - `build_stage/test/Dialect/FIRRTL/Reduction/port-pruner.mlir`
  - `build_stage/test/Dialect/HW/Reduction/hw-module-internal-name-sanitizer.mlir`
  - `build_stage/test/Dialect/HW/Reduction/hw-module-name-sanitizer.mlir`
  - `build_stage/test/Dialect/HW/Reduction/hw-sv-namehint-remover.mlir`
- Runtime python sanity:
  - `python3 -m py_compile lib/Dialect/ESI/runtime/python/esiaccel/codegen.py`
- ESI runtime integration test status:
  - `check-circt_integration-dialect-esi-runtime` could not run in this
    environment due lit configuration + missing `psutil` timeout support.

### Engineering Notes

- `ee4badcde` looked unpicked via patch-id, but functionality was already
  present in the local `circt-bmc` rewrite; re-picking only produced noisy
  conflicts.
- ImportVerilog queue/fork commits are high value but no longer "easy picks"
  on this tree due substantial local frontend/runtime divergence.

## Refresh (2026-02-28, follow-up pass)

- Current staging head: `77a0a4584`
- Stack size vs `origin/main`: `49` commits ahead

### Additional Picks In Follow-up Pass

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `2e3c1ea20` | `eee665fe1` | [ESI][Runtime] Support editable installs (#9764) |
| `9679a38c7` | `f13da0b62` | [PyRTG] Add String format function (#9762) |
| `b9249b45a` | `6c1f5affa` | [ESI][Runtime][Wheel] Don't re-export stdlib symbols |
| `77a0a4584` | `89778e1be` | [ImportVerilog] Fix silent failure on unsupported system calls (#9556) |

### Validation Added In Follow-up Pass

- Build:
  - `utils/ninja-with-lock.sh -C build_stage circt-translate`
- Python syntax sanity:
  - `python3 -m py_compile lib/Dialect/ESI/runtime/setup.py`
  - `python3 -m py_compile frontends/PyRTG/src/pyrtg/strings.py`
  - `python3 -m py_compile frontends/PyRTG/test/basic.py`
- ImportVerilog test status:
  - Direct lit run of `build_stage/test/Conversion/ImportVerilog/errors.sv`
    is not usable in this build config because `circt-translate` lacks
    `--import-verilog` in this environment.

### Deferred / Superseded In Follow-up Pass

- `e0b24742a` ([VerifToSMT] assertions in funcs): superseded by local newer
  behavior and tests; re-pick would regress.
- `b02e27691` ([FIRRTL] Chisel intrinsic format substitutions): empty
  cherry-pick after conflict resolution, already present functionally.
- `78ca033c2`, `6b8405a08` (ESI runtime pytest/integration test follow-ups):
  conflict with local removal/rework of integration test paths.
- `2fdae8fcf` ([ArcRuntime][VCDTrace] final timestep): conflicts with local
  Arc runtime file-layout divergence (upstream files removed/reworked here).
- `afcefdc19` ([comb] ~sext canonicalization) and `db50d3131` ([LLHD] mem2reg
  wait-live capture): re-attempted in this pass, both resolved to empty
  cherry-picks after conflict resolution (already functionally present).

## Refresh (2026-02-28, continued mining pass)

- Starting staging head for this pass: `a3bf77e8d`
- Current staging head after this pass: `9aaafd297`
- Stack size vs `origin/main`: `60` commits ahead

### Additional Picks In This Pass

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `0a9115330` | N/A (local follow-up) | [LTL] Fix PastOp optional-clock callsites |
| `6ed59274b` | `0f99432e5` | [Arc] Lower time operations to LLVM IR (#9757) |
| `a36b0236a` | `9b52782c1` | [FIRRTL] Lazily construct CircuitNamespace, NFC (#9767) |
| `79b817856` | `83ff24e5b` | [PyCDE] Remove python 3.8, 3.9 and add 3.14 builds (#9769) |
| `8910e519b` | `4f8fdaa73` | [PyCDE] Disable cocotb tests by default (#9770) |
| `9aaafd297` | `35e49085c` | [Synth] Add resource usage analysis (#9717) |

### Validation Added In This Pass

- Build:
  - `utils/ninja-with-lock.sh -C build_stage circt-opt`
  - `utils/ninja-with-lock.sh -C build_stage firtool`
  - `utils/ninja-with-lock.sh -C build_stage circt-opt circt-synth`
- Arc lowering sanity (manual lit-equivalent checks):
  - `build_stage/bin/circt-opt test/Conversion/ArcToLLVM/lower-arc-to-llvm.mlir --lower-arc-to-llvm`
  - Verified output contains `llvm.func @Time`, expected loads, and no residual `int_to_time`/`time_to_int` ops.
  - Verified existing `sim.proc.print`/`sim.terminate` lowerings still emit `printf`/`exit` calls.
- FIRRTL instance-choice symbol pass sanity:
  - `build_stage/bin/circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-populate-instance-choice-symbols))' test/Dialect/FIRRTL/populate-instance-choice-symbols.mlir`
  - Verified both expected `sv.macro.decl` entries and `instance_macro` attributes in output.
- firtool CLI sanity:
  - `build_stage/bin/firtool --help`
  - Verified `-j` alias and `--num-threads=<N>` option are present.
- PyCDE packaging sanity:
  - `python3 -m py_compile frontends/PyCDE/setup.py`
  - Parsed `frontends/PyCDE/pyproject.toml` via `tomllib`.
- Synth analysis sanity:
  - `build_stage/bin/circt-opt --synth-print-resource-usage-analysis='output-file="-"' test/Dialect/Synth/resource.mlir`
  - Verified expected top-level counts (`comb.and/or/xor`, `synth.aig.and_inv`).
  - `build_stage/bin/circt-synth test/circt-synth/path-e2e.mlir -analysis-output=- -top test -analysis-output-format=json`
  - Verified JSON output contains `"module_name":"test"`.

### Deferred / Superseded In This Pass

- Empty (already present functionally after conflict resolution):
  - `6e3d168f6`, `7a25c970c`, `49e7f3af0`, `6542026a9`,
    `11cc66244`, `a0ddbccf0`, `5ade31e47`, `debd22694`, `5f7d374a7`,
    `ec285f538`.
- Skipped due path/layout divergence:
  - `16b706093` (`TraceEncoder.h` modify/delete conflict; file path removed/reworked locally).

### Engineering Notes

- `check-circt` currently fails in this staging tree due an unrelated baseline
  AOT compile issue (`registerJITRuntimeSymbols` undeclared in
  `tools/circt-sim/AOTProcessCompiler.cpp`). This is outside the scope of the
  upstream picks above; targeted build/test validation was used instead.
- The Arc time-lowering pick was effectively already present in code; only a
  minor test-file whitespace delta remained after conflict reconciliation.

## Refresh (2026-02-28, RTG/ImportVerilog incremental pass)

- Starting staging head for this pass: `b13803546`
- Current staging head after this pass: `9c15977a3`
- Stack size vs `origin/main`: `70` commits ahead

### Additional Picks In This Pass

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `9e60e9805` | `353ac13ab` | [RTG][LinearScanRegisterAllocation] Fix nightly (#9728) |
| `9c15977a3` | `89778e1be` | [ImportVerilog] Fix silent failure on unsupported system calls (#9556) |

### Validation Added In This Pass

- Build:
  - `utils/ninja-with-lock.sh -C build_stage circt-opt`
- RTG-focused lit:
  - `/home/thomas-ahle/circt/build_test/bin/llvm-lit -sv build_stage/test/Dialect/RTG/Transform/linear-scan-register-allocation.mlir build_stage/test/Dialect/RTG/Transform/elaboration.mlir`
  - `/home/thomas-ahle/circt/build_test/bin/llvm-lit -sv build_stage/test/Dialect/RTG/IR/basic.mlir build_stage/test/Dialect/RTG/IR/canonicalization.mlir build_stage/test/circt-verilog/library-files.sv build_stage/test/circt-verilog/library-locations.sv build_stage/test/circt-verilog/command-files.sv`
- ImportVerilog `errors.sv` validation status:
  - `llvm-lit` invocation of `build_stage/test/Conversion/ImportVerilog/errors.sv` is not usable in this staging build because `build_stage/bin/circt-translate` lacks `--import-verilog`.
  - The regression test update was still retained; this environment mismatch is pre-existing and unrelated to the patch logic.

### Deferred / Superseded In This Pass

- Conflicted and skipped as superseded by newer local implementations:
  - `d3ddbe121` (queue `$` indexing), `79369384c` (`$sampled` result usability), `fd13bc452` (`NegRealOp` conversion test/addition), `e0b24742a` (`VerifToSMT` assertions-in-functions), `0f99432e5` (Arc time-op lowering).
- Attempted and resolved to empty cherry-picks (already present functionally):
  - `c1ef24514`, `11cc66244`, `ec285f538`, `a0b58ccce`, `7a25c970c`, `49e7f3af0`, `6e3d168f6`, `6542026a9`, `5f7d374a7`.
- Skipped due path/layout divergence:
  - `16b706093` (`TraceEncoder.h` modify/delete conflict; file layout diverged locally).

### Engineering Notes

- The current branch already carries a large portion of recent upstream fixes under different local commit hashes, so many `git cherry` `+` candidates become empty when re-applied.
- RTG cherry-picks remain the highest-signal/lowest-risk area for this tree at the moment; ImportVerilog queue/sampled series should be handled in a dedicated sync pass because local frontend changes are substantial.

## ImportVerilog Regression Coverage Sync (2026-02-28, queue/sampled follow-up)

- Starting staging head for this pass: `9c15977a3`
- Current staging head after this pass: pending commit

### Scope

- Follow-up to previously deferred upstream commits:
  - `d3ddbe121` (`$` literal in queue indexing)
  - `79369384c` (sampled-value results usable by Moore ops)
- Local tree already had the functional lowering support, but test coverage was missing for:
  - nested queue `$` indexing paths in `queues.sv`
  - direct `== $rose/$fell/$stable/$changed` assertion forms in `builtins.sv`

### Local Changes

- Added `QueueNestedDollarIndexTest` to `test/Conversion/ImportVerilog/queues.sv`.
- Added explicit sampled-value equality checks (`rose_eq`, `fell_eq`, `stable_eq`, `changed_eq`) to `test/Conversion/ImportVerilog/builtins.sv`.

### Validation Added In This Pass

- Targeted FileCheck + frontend parsing checks:
  - `/home/thomas-ahle/circt/build_test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/queues.sv | /home/thomas-ahle/circt/build_test/bin/FileCheck test/Conversion/ImportVerilog/queues.sv`
  - `/home/thomas-ahle/circt/build_test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/queues.sv`
  - `/home/thomas-ahle/circt/build_test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/builtins.sv | /home/thomas-ahle/circt/build_test/bin/FileCheck test/Conversion/ImportVerilog/builtins.sv`
  - `/home/thomas-ahle/circt/build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/builtins.sv`

### Engineering Notes

- The first version of the new `builtins.sv` checks was too strict about SSA/value ordering and assertion op flavor (`verif.clocked_assert` vs `verif.assert`), and failed.
- Checks were relaxed to assert semantic presence (`moore.int_to_logic`, `moore.eq`, labeled assert) without constraining unstable textual details.

## CI Concurrency Follow-up (2026-02-28)

- Starting staging head for this pass: `4169bfd64`
- Current staging head after this pass: `21ebc4af5`

### Additional Picks In This Pass

| Local commit | Upstream commit | Subject |
| --- | --- | --- |
| `21ebc4af5` | `a0b58ccce` | [CI] Cancel in-progress PR builds on new push (#9751) |

### Local Conflict Resolution/Adaptation

- Upstream commit touched four workflows, but this tree already had local concurrency tuning in two of them.
- Kept exactly one `concurrency` block per workflow and retained existing behavior where appropriate.
- Applied upstream PR-scoped cancellation semantics to:
  - `.github/workflows/buildAndTest.yml`
  - `.github/workflows/buildAndTestWindows.yml`
- Left existing local short-integration cancellation behavior unchanged (removing accidental duplicate block introduced during cherry-pick resolution).
- Removed duplicate `concurrency` block in `testPycdeESI.yml` generated during initial conflict application.

### Validation Added In This Pass

- Structural workflow sanity checks:
  - `rg -n "^concurrency:|cancel-in-progress|group:" .github/workflows/buildAndTest.yml .github/workflows/buildAndTestWindows.yml .github/workflows/shortIntegrationTests.yml .github/workflows/testPycdeESI.yml`
  - Verified one `concurrency` block per file and expected `cancel-in-progress` expressions.

### Deferred / Superseded In This Pass

- Skipped as effectively already present or incompatible with local evolved implementation:
  - `17330f8a9` (fork-join blocks; conflicts with current `ForkOp`/`ForkTerminatorOp` path)
  - `0f99432e5` (Arc time lowering; resolved to empty)
  - `afcefdc19` (Comb canonicalization; overlaps with local SextMatcher hardening and test evolution)
  - `2b9e89d41` (MooreToCore unreachable default removal; empty on this branch)
