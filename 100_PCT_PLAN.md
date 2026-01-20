# CIRCT circt-verilog 100% sv-tests Compliance Plan

## Executive Summary

| Metric | Current | Target |
|--------|---------|--------|
| Tests Passing | 498/725 | 725/725 |
| Pass Rate | 68.7% | 100% |
| Tests to Fix | 227 | 0 |

This plan organizes the 227 failing tests into 4 implementation phases, ordered by impact (tests unblocked) and complexity.

---

## Phase 1: High-Impact Quick Wins (+85 tests, targeting 80%)

### 1.1 UVM Streaming Operator Fix (63+ tests) - CRITICAL

**Error**: `lvalue streaming expected IntType, got '!moore.open_uarray<i1>'`

**Root Cause**: Streaming operators `{<<}` and `{>>}` don't handle `OpenUnpackedArrayType` in lvalue contexts, which UVM uses extensively for data packing.

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp` - Streaming operator handling (~line 3000+)

**Implementation**:
1. Extend lvalue streaming to accept `OpenUnpackedArrayType`
2. Add conversion logic to flatten open arrays to bit streams
3. Handle dynamic sizing through runtime bounds calculation

**Complexity**: Large
**Tests Unblocked**: 63+ (all UVM-based randomization tests in Chapter 18)
**Status**: DONE (Iteration 33)

---

### 1.2 File I/O System Tasks (15+ tests)

**Missing Functions**: `$fgetc`, `$fgets`, `$fread`, `$feof`, `$ferror`, `$fflush`, `$fstrobe`, `$fmonitor`, `$ungetc`, `$ftell`, `$dumpfile`, `$dumpvars`, `$dumpports`, `$strobe`, `$monitor`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp` - System call switch statement

**Implementation**:
1. Add case handlers for each file I/O function
2. Lower to `moore.call` or `sim` dialect operations
3. Handle file descriptor types and empty argument expressions

**Complexity**: Medium
**Tests Unblocked**: 15+ (Chapter 21)
**Status**: DONE (Iteration 33)

---

### 1.3 Bit Manipulation System Functions (7+ tests)

**Missing**: `$onehot`, `$onehot0`, `$countbits`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp` - System call handling

**Implementation**:
1. Add handlers using existing `comb` dialect popcount/comparison ops
2. `$onehot(x)` = `popcount(x) == 1`
3. `$onehot0(x)` = `popcount(x) <= 1`
4. `$countbits(x, ctrl)` = count bits matching control values

**Complexity**: Small
**Tests Unblocked**: 7+ (Chapter 20.9)

---

## Phase 2: Statement and Control Flow (+28 tests, targeting 84%)

### 2.1 Random Statements (14 tests)

**Errors**: `unsupported statement: RandCase`, `unsupported statement: RandSequence`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Statements.cpp`

**Implementation**:
1. Add `visitStmt(RandCaseStatement)` - weighted random selection
2. Add `visitStmt(RandSequenceStatement)` - production rule evaluation
3. Generate weighted random selection using `$urandom_range`

**Example transformation**:
```systemverilog
randcase
  3: x = 1;  // 30% probability
  7: x = 2;  // 70% probability
endcase
```
becomes conditional on `$urandom_range(0,9) < 3`

**Complexity**: Large
**Tests Unblocked**: 14 (Chapter 18.16, 18.17)
**Status**: PARTIAL (RandSequence supported; randcase supported; break/return supported; randjoin>1 uses fork/join)

---

### 2.2 Pattern Matching (8 tests)

**Errors**: `unsupported statement: PatternCase`, `unsupported expression: TaggedUnion`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Statements.cpp` - PatternCase
- `lib/Conversion/ImportVerilog/Expressions.cpp` - TaggedUnion

**Implementation**:
1. `PatternCaseStatement` → decompose into conditional checks
2. `TaggedUnion` expressions → struct with tag field + union data
3. Pattern matching → tag comparison + field extraction

**Complexity**: Large
**Tests Unblocked**: 8 (Chapters 7, 11, 12)
**Status**: PARTIAL (PatternCase + conditional matches done; structure/variable patterns still missing)

---

### 2.3 Control Flow in Initial Blocks (6 tests)

**Error**: `'seq.initial' op expects region #0 to have 0 or 1 blocks`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Statements.cpp`
- `lib/Conversion/MooreToCore/MooreToCore.cpp`

**Implementation**:
1. Structure control flow to avoid multi-block regions
2. Lower `return`/`break`/`continue` to structured control flow
3. Add CFG structurization pass if needed

**Complexity**: Medium
**Tests Unblocked**: 6 (Chapter 12.7, 12.8)

---

## Phase 3: Module and Class Features (+18 tests, targeting 87%)

### 3.1 Clocking Blocks (4-5 tests)

**Error**: `unsupported module member: ClockingBlock`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Structure.cpp`

**Implementation**:
1. Add `visitMember(ClockingBlock)` handler
2. Parse input/output skews as timing constraints
3. Handle `default clocking` and `global clocking` declarations
4. Lower to timing annotation attributes or dedicated ops

**Complexity**: Large
**Tests Unblocked**: 4-5 (Chapter 14)

---

### 3.2 Constraint Mode Override Fix (6+ tests)

**Error**: `cannot override built-in method 'constraint_mode'`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Structure.cpp` - Class method handling

**Implementation**:
1. Special-case `constraint_mode()`, `rand_mode()`, `randomize()` as built-in methods
2. Allow proper method resolution without override errors
3. Preserve built-in semantics while allowing calls

**Complexity**: Medium
**Tests Unblocked**: 6+ (Chapter 18.8, 18.9)

---

### 3.3 Distribution Functions (4+ tests)

**Error**: `unsupported expression: EmptyArgument`

**Missing**: `$dist_t`, `$dist_exponential`, `$dist_poisson`, `$dist_uniform`, `$dist_normal`, `$dist_erlang`, `$dist_chi_square`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp`

**Implementation**:
1. Handle empty/default arguments in system function calls
2. Add distribution function handlers
3. Implement using standard random algorithms (Box-Muller, inverse CDF, etc.)

**Complexity**: Medium
**Tests Unblocked**: 4+ (Chapter 20.15)
**Status**: DONE (Iteration 33)

---

## Phase 4: Array and Type System (+25 tests, targeting 90%+)

### 4.1 Dynamic Queue Operations (7 tests)

**Error**: `unbounded literal ($) used outside of queue/array context`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp`

**Implementation**:
1. Extend `$` literal handling to queue slice contexts
2. Support `q[0:$-1]` (all but last) and `q[1:$]` (all but first)
3. Handle queue concatenation: `q = {q, item}`, `q = {item, q}`

**Complexity**: Medium
**Tests Unblocked**: 7 (Chapter 7 queues)
**Status**: PARTIAL (queue concat runtime now implemented with element size; dynamic-array `$` indexing supported; queue slice `$` indices covered; `$` literal edge cases in slices pending)

---

### 4.2 Array Methods (8+ tests)

**Missing Methods**: `xor()`, `sum()`, `product()`, `and()`, `or()`, `shuffle()`, `rsort()`, `unique_index()`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp` - Array method calls

**Implementation**:
1. Reduction methods → loop with accumulator
2. `shuffle()` → Fisher-Yates algorithm
3. `rsort()` → sort with reverse comparator
4. `unique_index()` → track seen values, return indices of unique elements

**Complexity**: Medium
**Tests Unblocked**: 8+ (Chapter 7 arrays)

---

### 4.3 String Type Conversions (7 tests)

**Error**: `expression of type '!moore.string' cannot be cast to simple bit vector`

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Expressions.cpp` - Cast handling
- `lib/Conversion/ImportVerilog/Types.cpp`

**Implementation**:
1. Add string → bitvector conversion (8 bits per character)
2. Handle dynamic string length
3. Implement proper padding/truncation for fixed-width targets

**Complexity**: Medium
**Tests Unblocked**: 7

---

### 4.4 Array Comparison Legalization (2 tests)

**Error**: `failed to legalize operation 'moore.uarray_cmp'`

**Files to Modify**:
- `lib/Conversion/MooreToCore/MooreToCore.cpp`

**Implementation**:
1. Add lowering pattern for `moore.uarray_cmp`
2. Generate element-wise comparison loop
3. Reduce with AND for equality, lexicographic for ordering

**Complexity**: Small
**Tests Unblocked**: 2 (Chapter 7)

---

### 4.5 Interface Declaration (1 test)

**Files to Modify**:
- `lib/Conversion/ImportVerilog/Structure.cpp`

**Implementation**: Debug and fix edge case in interface instantiation

**Complexity**: Small
**Tests Unblocked**: 1 (Chapter 25)

---

## Remaining Tests (~71 tests)

After Phase 4, approximately 71 tests may still fail due to:
- Overlapping dependencies (UVM tests needing multiple features)
- Edge cases discovered during implementation
- Simulation-only features (not elaboration)

These should be addressed iteratively after the main phases.

---

## Implementation Schedule

| Phase | Focus | Tests Fixed | Cumulative Pass Rate |
|-------|-------|-------------|---------------------|
| 1 | Quick wins (streaming, file I/O, bit funcs) | +85 | 80.4% |
| 2 | Control flow (random stmts, patterns, jumps) | +28 | 84.3% |
| 3 | Module features (clocking, constraints, dist) | +18 | 86.8% |
| 4 | Type system (queues, arrays, strings) | +25 | 90.2% |
| 5 | Remaining fixes | +71 | 100% |

---

## Testing Strategy

### Per-Feature Testing
```bash
# Run single test
export PATH="$HOME/circt/build/bin:$PATH"
cd ~/sv-tests
./tools/runner --runner circt_verilog tests/chapter-X/test.sv
```

### Phase Validation
```bash
# Full suite after each phase
make tests RUNNERS_FILTER=circt_verilog -j$(nproc)
make report RUNNERS_FILTER=circt_verilog
```

### Regression Check
```bash
# Compare before/after
python3 -c "
import csv
with open('out/report/report.csv') as f:
    r = csv.DictReader(f)
    passed = sum(1 for x in r if x['Pass']=='True')
    print(f'Passed: {passed}/725')
"
```

---

## Key Files Reference

| File | Size | Purpose |
|------|------|---------|
| `lib/Conversion/ImportVerilog/Expressions.cpp` | 228KB | Expression lowering, system calls, streaming |
| `lib/Conversion/ImportVerilog/Statements.cpp` | 51KB | Statement lowering, control flow |
| `lib/Conversion/ImportVerilog/Structure.cpp` | 114KB | Module/class/interface structure |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | 367KB | Moore→Core dialect lowering |
| `include/Dialect/Moore/MooreOps.td` | 5430 lines | Moore operation definitions |

---

## Dependencies

```
Phase 1.1 (UVM Streaming) ──────┐
                                ├──> UVM Tests (63+)
Phase 2.1 (Random Stmts) ───────┤
                                │
Phase 3.3 (Distribution) ───────┘

Phase 2.2 (Pattern Match) ──────> Tagged Union Tests

Phase 1.2 (File I/O) ───────────> I/O Tests (standalone)

Phase 3.1 (Clocking) ───────────> Clocking Tests (standalone)
```

Most features are independent and can be parallelized across developers.

---

## Success Criteria

1. All 725 sv-tests pass with circt-verilog runner
2. No regressions in existing passing tests
3. Clean `make report` with 100% pass rate
4. HTML report at `out/report/index.html` shows full compliance
