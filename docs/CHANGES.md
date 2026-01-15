# Recent Changes (UVM Parity Work)

## January 15, 2026 - MILESTONE M1 ACHIEVED: UVM Parses with Zero Errors!

**Status**: UVM parsing complete! `uvm_pkg.sv` parses with **zero errors** (exit code 0). All 8 mbit/* AVIP global packages also parse successfully.

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
