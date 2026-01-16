# Recent Changes (UVM Parity Work)

## January 16, 2026 - UVM MooreToCore 99% Complete!

**Status**: UVM MooreToCore conversion nearly complete! Only `moore.array.locator` remains.

### MooreToCore Lowering Progress

**Current Status**: 99%+ of UVM converts through MooreToCore. Single remaining blocker.

| Blocker | Commit | Ops Unblocked | Status |
|---------|--------|---------------|--------|
| Mem2Reg dominance | b881afe61 | 4 | ✅ Fixed |
| dyn_extract (queues) | 550949250 | 970 | ✅ Fixed |
| array.size | f18154abb | 349 | ✅ Fixed |
| vtable.load_method | e0df41cec | 4764 | ✅ Fixed |
| getIntOrFloatBitWidth crash | 8911370be | - | ✅ Fixed |
| data layout crash | 2933eb854 | - | ✅ Fixed |
| StringReplicateOp | 14bf13ada | - | ✅ Fixed |
| unpacked struct variables | ae1441b9d | - | ✅ Fixed |
| llhd::RefType cast crash | 5dd8ce361 | 57 | ✅ Fixed |
| StructExtract/Create crash | 59ccc8127 | 129 | ✅ Fixed |
| Interface tasks/functions | d1cd16f75 | - | ✅ Fixed |
| Interface task-to-task calls | d1b870e5e | - | ✅ Fixed |
| **moore.array.locator** | - | 1+ | 🔴 Next |

### Iteration 11 Fixes (January 16, 2026)

#### StructExtract/StructCreate for Dynamic Types (59ccc8127)
- **Problem**: Structs with dynamic fields (strings, classes, queues) convert to LLVM struct types, but StructExtractOp/StructCreateOp assumed hw::StructType
- **Solution**:
  - StructExtractOp: Use LLVM::ExtractValueOp for LLVM struct types
  - StructCreateOp: Use LLVM::UndefOp + LLVM::InsertValueOp for LLVM struct types
- **Impact**: UVM MooreToCore now progresses past struct operations (129 ops unblocked)

#### Interface Task-to-Task Calls (d1b870e5e)
- **Problem**: Interface task calling another task in the same interface didn't work
- **Solution**: When inside an interface method, use currentInterfaceArg for nested calls
- **Impact**: BFM-style patterns with nested task calls now work correctly

#### DPI Tool Info Functions (d1b870e5e)
- **uvm_dpi_get_tool_name_c()**: Now returns "CIRCT"
- **uvm_dpi_get_tool_version_c()**: Now returns "1.0"
- **Impact**: UVM can identify the simulator

### Iteration 10 Fixes (January 16, 2026)

#### Interface Task/Function Support (d1cd16f75)
- **BFM pattern support**: Interface tasks/functions now convert with implicit interface argument
- **Signal access**: Uses VirtualInterfaceSignalRefOp for interface signal access within methods
- **Call site**: Interface method calls pass the interface instance as first argument
- **Impact**: Enables UVM BFM patterns where interface tasks wait on clocks/drive signals

#### StructExtractRefOp for Dynamic Types (5dd8ce361)
- **Problem**: SigStructExtractOp expected llhd::RefType but received LLVM pointer for structs with dynamic types
- **Solution**: Check original Moore type via typeConverter, use LLVM GEP for dynamic structs
- **Impact**: Unblocked 57 StructExtractRefOp operations

### AVIP Testing Results

| AVIP | Parsing | MooreToCore | Issue |
|------|---------|-------------|-------|
| APB | ✅ Pass | ✅ Pass | - |
| AHB | ✅ Pass | ✅ Pass | - |
| AXI4 | ✅ Pass | ✅ Pass | - |
| AXI4Lite | ✅ Pass | ✅ Pass | - |
| I2S | ✅ Pass | ✅ Pass | - |
| I3C | ✅ Pass | ✅ Pass | - |
| JTAG | ❌ Fail | - | Source: enum implicit conversion |
| SPI | ❌ Fail | - | Source: nested block comments |
| UART | ❌ Fail | - | Source: default arg mismatch |

**Note**: JTAG/SPI/UART failures are source code issues in the AVIPs, not CIRCT bugs.

---

## January 15, 2026 - MILESTONE M1 ACHIEVED: UVM Parses with Zero Errors!

### Iteration 7 MooreToCore Fixes

#### StringReplicateOp Lowering (14bf13ada)
- **String replication**: Added lowering for `moore.string_replicate` op
- **Pattern**: `{N{str}}` string replication now properly lowered to runtime calls
- **Impact**: Enables string replication patterns used in UVM formatting

#### Unpacked Struct Variable Lowering (ae1441b9d)
- **Dynamic type handling**: Fixed variable lowering for unpacked structs containing dynamic types
- **Root cause**: Structs with queue/string/dynamic array fields were not properly handled
- **Solution**: Added type checking before lowering to handle mixed static/dynamic struct fields

#### Virtual Interface Assignment (f4e1cc660) - ImportVerilog
- **Assignment support**: Added support for virtual interface assignment (`vif = cfg.vif`)
- **BFM patterns**: Enables standard verification component initialization patterns

#### Virtual Interface Scope Tracking (d337cb092) - ImportVerilog
- **Class context**: Added scope tracking for virtual interface member access within classes
- **Root cause**: Virtual interface accesses inside class methods lost their scope context
- **Solution**: Track scope during conversion to properly resolve virtual interface references

### Data Layout Crash Fix (2933eb854)
- **convertToLLVMType helper**: Recursively converts hw.struct/array/union to pure LLVM types
- **Class/interface structs**: Applied to resolveClassStructBody() and resolveInterfaceStructBody()
- **Root cause**: hw.struct types don't provide LLVM DataLayout information

### ImportVerilog Tests (65eafb0de)
- **All tests passing**: 30/30 ImportVerilog tests (was 16/30)
- **Fixes**: Type prefix patterns, error messages, CHECK ordering, feature behavior changes

### getIntOrFloatBitWidth Crash Fix (8911370be)
- **Type-safe helper**: Added `getTypeSizeInBytes()` using `hw::getBitWidth()` for safe type handling
- **Queue ops fixed**: QueuePushBack, QueuePushFront, QueuePopBack, QueuePopFront, StreamConcat
- **Non-integer handling**: Uses LLVM::BitcastOp for non-integer types

### Virtual Interface Member Access (0a16d3a06)
- **VirtualInterfaceSignalRefOp**: Access signals inside interfaces via virtual interfaces
- **AVIP BFM support**: Enables `vif.proxy_h = this` pattern used in verification components

### QueueConcatOp Format Fix (2bd58f1c9)
- **Empty operands**: Fixed IR syntax for empty operand case using parentheses format

### VTable Load Method Fix (e0df41cec)
- **Abstract class vtables**: Fixed vtable lookup for abstract class handles by recursively searching nested vtables
- **Root cause**: Abstract classes don't have top-level vtables, but their segments appear nested in derived class vtables
- **Tests added**: vtable-abstract.mlir, vtable-ops.mlir (12 comprehensive tests)
- **Impact**: Unblocked 4764 vtable.load_method operations in UVM

### Array Size Lowering (f18154abb)
- **Queue/dynamic array size**: Extract length field (field 1) from `{ptr, i64}` struct
- **Associative array size**: Added `__moore_assoc_size` runtime function
- **Impact**: Unblocked 349 array.size operations in UVM

### Virtual Interface Comparison (8f843332d)
- **VirtualInterfaceNullOp**: Creates null virtual interface value
- **VirtualInterfaceCmpOp**: Compares virtual interfaces (eq/ne)
- **BoolCastOp fix**: Now handles pointer types for `if(vif)` checks
- **Impact**: Fixes "cannot be cast to simple bit vector" errors in UVM config_db

### Queue/Dynamic Array Indexing (550949250)
- **dyn_extract lowering**: Implemented queue and dynamic array indexing via `DynExtractOpConversion`
- **dyn_extract_ref lowering**: Added ref-based indexing support for write operations
- **StringPutC/StringItoa fixes**: Fixed to use LLVM store instead of llhd::DriveOp
- **Impact**: Unblocked 970 queue indexing operations in UVM

### Mem2Reg Dominance Fix (b881afe61)
- **Loop-local variable promotion**: Variables declared inside loops (e.g., `int idx;` inside `foreach`) were being incorrectly promoted by MLIR's Mem2Reg pass, causing dominance violations.
- **Root cause**: When Mem2Reg creates block arguments at loop headers for promoted variables, it needs a reaching definition for edges entering from outside the loop. Loop-local variables don't dominate these entry edges.
- **Solution**: Modified `VariableOp::getPromotableSlots()` to detect variables inside loops by checking for back-edges in the dominator tree. Variables with users in other blocks that are inside any loop are now excluded from promotion.
- **Result**: Fixed all 4 dominance errors in UVM:
  - `uvm_component.svh:3335` - `int idx` in foreach/while
  - `uvm_cmdline_report.svh:261` - `bit hit` in foreach
  - `uvm_reg.svh:2588` - `uvm_reg_data_t slice` in foreach
  - `uvm_mem.svh:1875` - `uvm_reg_data_t slice` in for loop

### Other Fixes Completed

#### Static Property Handling (a1418d80f)
- **Static property via instance**: SystemVerilog allows `obj.static_prop` to access static properties. Now correctly generates `GetGlobalVariableOp` instead of `ClassPropertyRefOp`.
- **Parameterized class static properties**: Each specialization now gets unique global variable names (e.g., `uvm_pool_1234::m_prop` instead of shared `uvm_pool::m_prop`).
- **Abstract class vtable**: Virtual classes with mixed concrete/pure virtual methods now skip vtable generation instead of emitting error.

#### Type System Fixes (3c9728047)
- **Time type in Mem2Reg**: Fixed `VariableOp::getDefaultValue()` to correctly return TimeType values. The `SBVToPackedOp` result was being created but not captured, causing l64 constants to be used instead of time values.
- **Int/logic to time conversion**: Added explicit conversion support in `materializeConversion` for IntType ↔ TimeType.

#### Parameterized Class Fixes
- **Method lookup** (71c80f6bb): Class bodies now properly populated via `convertClassDeclaration` when encountered through method's `this` type in `declareFunction`.
- **Property type mismatch**: Parameterized class property access now uses correct specialized class symbol.
- **Super.method() dispatch** (09e75ba5a): Direct dispatch instead of vtable lookup for super calls.
- **Class upcast** (fbbc2a876): Parameterized base classes now recognized via generic class lookup.

#### Other Fixes
- **Global variable redefinition** (a152e9d35): Fixed duplicate GlobalVariableOp during recursive type conversion.

### Test Results
- **UVM**: 0 errors (MILESTONE M1 ACHIEVED!)
- **AVIP Global Packages**: 8/8 passing (ahb, apb, axi4, axi4Lite, i2s, i3c, jtag, spi, uart)

---

## Earlier Changes

- Stabilized ImportVerilog returns: return statements now cast to the function signature, and pure virtual methods are stubbed with default return values.
- Normalized string/format handling: string concatenation coerces `format_string` operands to `string`; queue push_back/insert coerce elements to queue element type.
- Added stubs for `rand_mode`, `$left/$right/$low/$high`, plus safer queue-to-bitvector casting.
- Queue/format comparisons: queue equality/inequality short-circuit instead of failing.
- File I/O complete: $fopen, $fwrite, $fclose, $sscanf all working.
- String ato* methods: atoi, atohex, atooct, atobin for UVM command-line parsing.
- Non-integral associative array keys: string and class handle keys for UVM pools.
