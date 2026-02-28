# Upstream Cherry-Pick Log

## 2026-02-28 (session: post-issue-8-9 sweep)

- Fork base: `67ce82812` (`origin/main`)
- Upstream head scanned: `5f7d374a7` (`upstream/main`)

### Applied

1. Upstream `c1ef24514` -> local `9b73dc773`
   - Subject: `[Comb] Fix SextMatcher crash on block arguments`
   - Why: crash fix (block-argument matcher safety)
   - Validation:
     - `llvm-lit --filter 'Dialect/Comb/canonicalization\\.mlir'` (pass)

2. Upstream `49e7f3af0` -> local `6ffdcaceb`
   - Subject: `[HW] Make sure the index type for arrays is at least i1`
   - Why: prevents `i0` index generation and downstream crashes
   - Validation:
     - `llvm-lit --filter 'Dialect/HW/hw-convert-bitcasts\\.mlir'` (pass)

3. Upstream `52b012089` -> local `50559f46d`
   - Subject: `[ConvertToLLVM] Add hw::ConstantOp conversion support and tests`
   - Why: enforces legal conversion pipeline and keeps `hw.constant` from leaking
   - Validation:
     - `llvm-lit --filter 'Conversion/CombToLLVM/comb-to-llvm\\.mlir'` (pass)

4. Upstream `7a25c970c` -> local `17a7ee22a`
   - Subject: `[HWToLLVM] Take the correct data layout alignment for alloca`
   - Why: correctness/runtime safety fix for generated LLVM with stricter alignment
   - Validation:
     - `llvm-lit --filter 'Conversion/HWToLLVM/spill_alignment\\.mlir'` (pass)

### Attempted / Deferred

1. Upstream `f4e0bd1d3` (`[ImportVerilog] Pass library files to slang`)
   - Status: deferred
   - Reason: in this fork, an equivalent library-file handling path already exists;
     naive cherry-pick duplicates loading and breaks `-l<lib>=<file>` behavior.

2. Upstream `d3ddbe121` (`[ImportVerilog] Support '$' literal within queue indexing expressions`)
   - Status: deferred
   - Reason: high-conflict cherry-pick across Moore/ImportVerilog/MooreToCore and
     baseline tests; not an easy pick for this sweep.

### Next Start Point

- Continue from the same upstream scan window near:
  - `17330f8a9` `[Moore][ImportVerilog] Add support for fork-join blocks`
  - `79369384c` `[ImportVerilog] Make sampled value functions' results usable by Moore ops`
  - `346a108ea` `[LTL][ImportVerilog] Add support for $sampled`
