# Iteration 45: DPI-C Import Stubs for UVM - Summary

## Goal
Implement DPI-C import support so UVM runtime functions work, eliminating "call skipped" remarks and providing functional stubs for UVM's external dependencies.

## Problem
UVM compilation showed numerous remarks like:
```
remark: DPI-C imports not yet supported; call to 'uvm_hdl_deposit' skipped
remark: DPI-C imports not yet supported; call to 'uvm_re_compexecfree' skipped
```

UVM depends on DPI-C functions for:
- **HDL access**: `uvm_hdl_deposit`, `uvm_hdl_force`, `uvm_hdl_release_and_read`, etc.
- **Regex**: `uvm_re_comp`, `uvm_re_exec`, `uvm_re_free`, `uvm_re_buffer`, etc.
- **Command line**: `uvm_dpi_get_next_arg_c`, `uvm_dpi_get_tool_name_c`, `uvm_dpi_get_tool_version_c`

## Solution

### 1. Runtime Stub Functions (MooreRuntime)

**Added to `/home/thomas-ahle/circt/include/circt/Runtime/MooreRuntime.h`:**
- HDL access function declarations (check_path, deposit, force, release, read)
- Regular expression function declarations (comp, exec, free, buffer, compexecfree, deglobbed)
- Command line / tool info function declarations (get_next_arg, get_tool_name, get_tool_version)

**Implemented in `/home/thomas-ahle/circt/lib/Runtime/MooreRuntime.cpp`:**

#### HDL Access Stubs
- `uvm_hdl_check_path()` - Always returns success (path exists)
- `uvm_hdl_deposit()` - Stores values in internal map for later retrieval
- `uvm_hdl_force()` - Marks values as forced in internal map
- `uvm_hdl_release_and_read()` - Releases force and returns stored value
- `uvm_hdl_release()` - Clears forced flag
- `uvm_hdl_read()` - Returns stored value or 0

#### Regular Expression Stubs
- `uvm_re_comp()` - Creates regex stub object (simplified substring matching)
- `uvm_re_exec()` - Performs substring search (not full regex)
- `uvm_re_free()` - Frees regex stub object
- `uvm_re_buffer()` - Returns last match buffer
- `uvm_re_compexecfree()` - Combined compile/exec/free in one call
- `uvm_re_deglobbed()` - Converts glob patterns to regex format

**Note**: Full regex implementation would require PCRE2 or similar library. Current implementation uses simplified substring matching since `std::regex` requires exception support which is disabled (`-fno-exceptions`).

#### Command Line / Tool Info Stubs
- `uvm_dpi_get_next_arg_c()` - Returns empty (no arguments)
- `uvm_dpi_get_tool_name_c()` - Returns "CIRCT"
- `uvm_dpi_get_tool_version_c()` - Returns "1.0"

### 2. ImportVerilog Changes

**Modified `/home/thomas-ahle/circt/lib/Conversion/ImportVerilog/Expressions.cpp`:**

**Before:**
```cpp
if (subroutine->flags & slang::ast::MethodFlags::DPIImport) {
  mlir::emitRemark(loc) << "DPI-C imports not yet supported; call to '"
                        << subroutine->name << "' skipped";
  // ... return dummy values inline ...
  return dummyValue; // Early return prevented actual call generation
}
```

**After:**
```cpp
if (subroutine->flags & slang::ast::MethodFlags::DPIImport) {
  mlir::emitRemark(loc) << "DPI-C import '" << subroutine->name
                        << "' will use runtime stub (link with MooreRuntime)";
  // Fall through to normal call generation below
}
```

**Key change**: Removed early return with dummy values, allowing normal function call generation to proceed. DPI functions are now declared as `func.func private @name` and called via `func.call @name`, which will link to MooreRuntime stubs at compile time.

### 3. Test Files

Created two comprehensive test files:

**`test/Conversion/ImportVerilog/dpi_imports.sv`:**
- Tests various DPI data types (int, string, chandle, void)
- Verifies function declarations are generated
- Verifies function calls are generated (not dummy values)
- Checks proper remarks are emitted

**`test/Conversion/ImportVerilog/uvm_dpi_basic.sv`:**
- Tests UVM-specific DPI functions
- HDL access: `uvm_hdl_deposit`
- Regex: `uvm_re_comp`, `uvm_re_exec`, `uvm_re_free`, `uvm_re_compexecfree`
- Tool info: `uvm_dpi_get_tool_name_c`
- Uses FileCheck to verify proper IR generation

Both tests **PASS**.

## Results

### Before
```
remark: DPI-C imports not yet supported; call to 'uvm_hdl_deposit' skipped
```
- DPI functions returned inline dummy values
- No actual function calls generated
- Functions couldn't be linked with external implementations

### After
```
remark: DPI-C import 'uvm_hdl_deposit' will use runtime stub (link with MooreRuntime)
```
- DPI functions generate proper `func.call` operations
- Function declarations created as `func.func private`
- Stubs provided by MooreRuntime library
- Ready for future replacement with full implementations

### UVM Compilation
```bash
cd ~/circt/build && ./bin/circt-verilog --ir-moore -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv
```

Now shows 12 DPI import remarks (all with "will use runtime stub" message):
- uvm_dpi_get_next_arg_c
- uvm_re_compexecfree
- uvm_re_buffer
- uvm_dpi_get_tool_name_c
- uvm_dpi_get_tool_version_c
- uvm_re_deglobbed
- uvm_hdl_deposit
- uvm_hdl_force
- uvm_hdl_release_and_read
- uvm_re_comp
- uvm_re_exec
- uvm_re_free

All generate proper function calls instead of being skipped.

## Files Modified

1. **`include/circt/Runtime/MooreRuntime.h`**
   - Added DPI-C function declarations (HDL, regex, command line)
   - ~120 lines added

2. **`lib/Runtime/MooreRuntime.cpp`**
   - Implemented DPI-C stub functions
   - ~310 lines added
   - Fixed forward declaration issue for `readElementValueUnsigned`

3. **`lib/Conversion/ImportVerilog/Expressions.cpp`**
   - Changed DPI handling from inline dummy values to actual function calls
   - Updated remark message to indicate runtime stub usage
   - ~10 lines changed

## Files Created

1. **`test/Conversion/ImportVerilog/dpi_imports.sv`**
   - General DPI-C import test
   - Tests multiple data types and function signatures

2. **`test/Conversion/ImportVerilog/uvm_dpi_basic.sv`**
   - UVM-specific DPI function test
   - Verifies proper call generation for UVM functions

## Technical Notes

### Exception Handling
- MooreRuntime builds with `-fno-exceptions` (LLVM/MLIR standard)
- Cannot use `std::regex` which throws exceptions on compilation errors
- Implemented simplified substring matching for regex stubs
- Full implementation would use PCRE2 or RE2 library

### HDL Access Implementation
- Maintains internal `std::unordered_map` for signal values
- Thread-safe with `std::mutex`
- Supports force/release semantics
- Returns stored values or defaults

### Future Improvements

1. **Full Regex Support**: Integrate PCRE2 library for proper regex matching
2. **Command Line Args**: Parse actual program arguments in `uvm_dpi_get_next_arg_c`
3. **HDL Hierarchy**: Connect to actual design hierarchy for `uvm_hdl_*` functions
4. **Performance**: Optimize HDL value storage for large designs
5. **Debugging**: Add optional debug prints (currently commented out)

## Build & Test

```bash
# Build
cd ~/circt/build
ninja circt-verilog MooreRuntime

# Test
./bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/dpi_imports.sv | FileCheck test/Conversion/ImportVerilog/dpi_imports.sv
./bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/uvm_dpi_basic.sv | FileCheck test/Conversion/ImportVerilog/uvm_dpi_basic.sv

# Test with UVM
./bin/circt-verilog --ir-moore -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv 2>&1 | grep "DPI-C import"
```

## Conclusion

✅ **Goal Achieved**: DPI-C imports now generate proper function calls with runtime stubs
✅ **UVM Support**: All UVM DPI functions have stub implementations
✅ **Tests Pass**: Both new test files pass FileCheck validation
✅ **Extensible**: Stubs can be replaced with full implementations in the future

The DPI-C infrastructure is now in place, allowing UVM code to compile and link successfully while providing a clear path for future enhancement with complete implementations.
