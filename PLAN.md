# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 9, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 200/200 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 696/696 eligible pass+xfail | 100% |
| sv-tests xfail (runtime gaps) | 41 | See breakdown below |
| sv-tests compile-only | 73 | Class-only + SVA UVM |
| AVIPs passing | 9/9 | APB, AHB, UART, I2S, I3C, SPI, AXI4, AXI4Lite, JTAG |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |

### xfail Breakdown (41 tests)

| Category | Count | Tests | Difficulty |
|----------|-------|-------|------------|
| Constraint inheritance | 1 | 18.5.2 | Medium |
| Distribution constraints | 1 | 18.5.4 | Medium |
| Foreach/array-reduction constraints | 2 | 18.5.8.* | Hard |
| Global constraints | 1 | 18.5.9 | Hard |
| Static constraint blocks | 1 | 18.5.11 | Medium |
| Functions in constraints | 1 | 18.5.12 | Hard |
| Soft constraint priorities | 2 | 18.5.14.1 | Medium |
| Infeasible constraint detection | 2 | 18.6.3 | Hard |
| Inline constraints (randomize with) | 4 | 18.7.* | Medium-Hard |
| Inline variable/constraint checker | 4 | 18.11.* | Medium-Hard |
| rand_mode static sharing | 1 | 18.8._3 | Medium |
| Random stability/seeding | 10 | 18.13-18.15 | Medium |
| UVM stream_unpack | 6 | uvm_agent_*, uvm_monitor_*, uvm_scoreboard_* | Hard |
| UVM resource_db | 1 | uvm_resource_db_read_by_name | Medium |
| ConstraintExprOp extraction gaps | 4 | (hidden in above) | Medium |

## Workstreams

### Track 1: Constraint Solver Improvements
**Goal**: Fix the constraint extraction pipeline and add missing constraint features.

**Root Cause Analysis**: The `extractRangeConstraints` pipeline has fundamental gaps:
1. `getPropertyName()` doesn't look through `zext`/`sext` (slang widens narrow types to i32)
2. Compound expressions (`and(uge, ule)` from `inside {[a:b]}`) aren't decomposed
3. `ConstraintInsideOp` is never created by ImportVerilog (dead code) - slang compiles `inside` to comparison trees
4. Constraint extraction only iterates current class body, not parent classes (no inheritance)
5. Cross-variable constraints (`b2 == b`) silently skipped (both sides are property reads)

**Tasks (priority order)**:
1. **Fix `getPropertyName()` to look through zext/sext** - Fixes ConstraintExprOp extraction for narrow types
2. **Decompose `and(cmp, cmp)` compound constraints** - Handles `inside {[a:b]}` patterns
3. **Walk parent class hierarchy for constraint inheritance** - `classDecl.getBaseAttr()` chain
4. **Add ConstraintExprOp extraction to inline constraint regions** - Fixes 18.7.* tests
5. **Implement cross-variable constraint solving** - `b2 == b` patterns (rejection sampling or copy-after)
6. **Distribution constraint extraction from ConstraintExprOp** - Currently only from ConstraintDistOp

**Tests affected**: ~12 tests could be fixed

### Track 2: Random Stability & Seeding
**Goal**: Implement deterministic random number generation per IEEE 1800-2017 §18.13-18.14.

**Current state**: `randomize()` uses a global mt19937 RNG. No per-object or per-thread seeding.

**Tasks**:
1. **Per-object RNG state** - Each class instance gets its own RNG seed based on object creation order
2. **`srandom(seed)` method** - Set object-specific RNG seed
3. **`get_randstate()` / `set_randstate()`** - Save/restore RNG state as string
4. **Thread stability** - Fork contexts get isolated RNG state
5. **Manual seed in `randomize(seed)`** - Override RNG for one call

**Tests affected**: 10 tests (18.13.3-5, 18.14.*, 18.15.*)

### Track 3: UVM Testbench Fixes
**Goal**: Fix the remaining 7 UVM testbench failures.

**Stream_unpack (6 tests)**:
- Error: `interpretOperation failed for process 3` on `moore.stream_unpack`
- Root cause: `StreamUnpackOpConversion` in MooreToCore doesn't handle `!moore.ref<open_uarray<i1>>` type
- These tests use UVM infrastructure with virtual interface patterns
- Virtual interface clock connections are one-time stores, not continuous drives

**resource_db (1 test)**:
- `uvm_resource_db::read_by_name` fails - the interceptor needs to be extended

**Tasks**:
1. **Fix StreamUnpackOp for open_uarray<i1> type** - Add type handling in MooreToCore conversion
2. **Fix virtual interface clock propagation** - Continuous drive instead of one-time store
3. **Extend resource_db interceptor** - Support `read_by_name` pattern

**Tests affected**: 7 tests

### Track 4: Advanced Constraint Features
**Goal**: Implement harder constraint features needed for complex UVM testbenches.

**Tasks**:
1. **Foreach iterative constraints** (18.5.8.1) - Loop over array elements with constraints
2. **Array reduction constraints** (18.5.8.2) - `array.sum()` in constraint context
3. **Global constraints** (18.5.9) - Cross-object constraint sharing
4. **Functions in constraints** (18.5.12) - Evaluate function calls during solving
5. **Infeasible constraint detection** (18.6.3) - Return 0 from randomize() when unsatisfiable
6. **Static constraint block sharing** (18.5.11) - Share constraint state across instances

**Tests affected**: ~8 tests

## Priority Matrix

| Priority | Track | Next Task | Impact |
|----------|-------|-----------|--------|
| P0 | Track 1 | Fix getPropertyName zext/sext + compound decomposition | Unblocks ~5 tests |
| P0 | Track 1 | Constraint inheritance (parent walking) | Unblocks 1 test + AVIP correctness |
| P1 | Track 2 | Per-object RNG state + srandom | Unblocks 5 tests |
| P1 | Track 3 | StreamUnpackOp type fix | Unblocks 6 UVM tests |
| P2 | Track 1 | Inline constraint ConstraintExprOp extraction | Unblocks 4 tests |
| P2 | Track 4 | Distribution from ConstraintExprOp | Unblocks 1 test |
| P3 | Track 4 | Foreach/global/functions constraints | Hard, 4 tests |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 200+ pass |
| ImportVerilog | `python3 build/bin/llvm-lit test/Conversion/ImportVerilog/ -v` | 268 pass |
| sv-tests sim | `bash utils/run_sv_tests_circt_sim.sh ~/sv-tests` | 0 fail, 0 timeout |
| AVIPs | `circt-sim X.mlir --top Y --max-time=500000000` | All 9 exit 0 |
| sv-tests BMC | `BMC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_bmc.sh ~/sv-tests` | 26/26 |
| sv-tests LEC | `LEC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_lec.sh ~/sv-tests` | 23/23 |

## Test Suites

- `~/sv-tests/` - IEEE 1800 compliance (750+ tests)
- `~/mbit/*avip*/` - AVIP testbenches (9 protocols)
- `~/verilator-verification/` - Verilator reference tests
- `~/yosys/tests/` - Yosys SVA tests
- `~/opentitan/` - OpenTitan formal (Codex handles)
