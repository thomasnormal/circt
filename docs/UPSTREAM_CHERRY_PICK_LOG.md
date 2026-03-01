# Upstream Cherry-Pick Log

This file tracks upstream `llvm/circt` mining so we can resume quickly without
re-triaging the same commits.

## Snapshot

- Date: 2026-02-28
- Local base used for mining: `origin/main` @ `9f24dc1d7`
- Upstream tip at time of scan: `upstream/main` @ `0f99432e5`
- Current staging head: `staging-upstream-easy-picks` @ `0f1439c64`
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
| `29f81dbaa` | `debd22694` | [FIRRTL] Add conservative IMDCE handling for InstanceChoiceOp (#9710) |
| `9010819e8` | `5ade31e47` | [FIRRTL] Support FInstanceLike operations in ModuleInliner (#9688) |
| `228e21581` | `70d66d7f4` | [FIRRTL][LowerToHW] Add InstanceChoiceOp lowering, Part 1 (#9742) |
| `0cbd42b69` | `274eeb55d` | [FIRRTL] Add domain create op (#9774) |
| `9c25e1e46` | `3d5455330` | [FIRRTL] Add instance macro attribute to InstanceChoice for Lowering (#9760) |
| `0f1439c64` | `15f3650af` | [FIRRTL] Change FInstanceLike to consider multiple referred modules (#9676) |

## Integration Status

- `staging-upstream-easy-picks` has not been merged to `main` yet.
- FIRRTL instance-choice picks are partially validated: build is green, but the
  focused `lit` rerun after `0f1439c64` is still pending.
- Last observed failures before `0f1439c64` were:
  - `test/Dialect/FIRRTL/inliner.mlir`
  - `test/Dialect/FIRRTL/imdce.mlir`
  with `firrtl.instance_choice` symbol/result verification errors.

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

Command to inspect what is still staging-only (not yet on `main`):

```bash
git log --oneline --no-merges main..staging-upstream-easy-picks
```

## 2026-03-01 Mining Pass (Main)

- Local base at start: `origin/main` @ `fd43fe5f2`
- Upstream tip scanned: `upstream/main` @ `5f7d374a7`

### Picked to `main`

| Local commit | Upstream commit | Subject | Notes |
| --- | --- | --- | --- |
| `0cf5a81a3` | `a0b58ccce` | [CI] Cancel in-progress PR builds on new push (#9751) | Applied cleanly; enables PR concurrency cancellation across 4 workflows. |
| `b0ecf17cd` | `8a3634d80` | [ImportVerilog] Add support for $swrite (#9782) | Applied with one test conflict in `test/Conversion/ImportVerilog/basic.sv`; kept both formatted and no-format `$swrite` coverage. |
| `9a51e183f` | `f4e0bd1d3` | [ImportVerilog] Pass library files to slang (#9680) | Applied cleanly. Added local regression `test/Tools/circt-verilog/library-files.test` plus inputs to lock behavior. |
| `559c7e278` | `11cc66244` | [Datapath] Bug Fix for Sign-Extension Logic when Lowering Partial Products to Booth Arrays (#9726) (#9744) | Applied cleanly in implementation; adjusted FORCE-BOOTH check order in local test to match canonicalization ordering, then validated `test/Conversion/DatapathToComb/datapath-to-comb.mlir`. |

### Attempted But Deferred

| Upstream commit | Subject | Outcome |
| --- | --- | --- |
| `d3ddbe121` | [ImportVerilog] Support `$` literal within queue indexing expressions (#9719) | Could not apply in this tree because `lib/Conversion/ImportVerilog/ImportVerilogInternals.h` has local unstaged changes from concurrent work. Retry when that file is clean or isolated. |
| `17330f8a9` | [Moore][ImportVerilog] Add support for fork-join blocks (#9682) | Cherry-pick produced semantic conflict in `lib/Conversion/ImportVerilog/Statements.cpp`; local implementation is already substantially diverged/richer. Aborted pick to avoid regression. |
| `3e24b7764` | [ImportVerilog] Fix $changed PastOp creation (#9652) | Cherry-pick conflicts with current assertion lowering architecture (`convertAssertionSystemCallArity1` fastpaths intentionally return `{}`; handling moved elsewhere). Aborted to avoid semantic regression; revisit only with full assertion-path audit. |
| `79369384c` | [ImportVerilog] Make sampled value functions' results usable by Moore ops | Cherry-pick conflicts with current sampled-value lowering and updated `builtins.sv` expectations (our tree uses different lowering path with `moore.past` checks). Aborted pending a targeted semantic audit instead of mechanical porting. |

### Resume Pointer

For the next pass from `main`, start with ImportVerilog candidates that are not blocked
by concurrent local edits:

1. `d3ddbe121` after `ImportVerilogInternals.h` is clean.
2. `17330f8a9` only if we intentionally reconcile fork/join lowering semantics in `Statements.cpp`.
3. non-ImportVerilog bug-fix candidates in clean files (e.g. Arc/FIRRTL/HW) when local frontend files remain actively edited.
