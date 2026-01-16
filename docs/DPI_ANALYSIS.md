# DPI-C Analysis for UVM Runtime Support

This document analyzes the DPI-C (Direct Programming Interface) imports used by UVM and what is needed for actual runtime execution in CIRCT.

## Summary

UVM uses 22 DPI-C imported functions across 4 categories:
1. **Regex/Glob Matching** (8 functions) - CRITICAL for UVM resource/factory lookup
2. **HDL Backdoor Access** (6 functions) - Required for register access
3. **Command-line Processing** (3 functions) - Required for plusargs
4. **Polling/Callbacks** (6 functions) - Optional, for advanced signal monitoring

## Current CIRCT Status

- **ImportVerilog**: DPI-C imports are parsed and declared as `func.func private` operations
- **Runtime**: DPI calls return stub default values (0 for int, "" for string, no-op for void)
- **MooreToCore**: No DPI-specific lowering yet; calls would need runtime library linkage

## DPI Function Categories

### 1. Regex/Glob Matching (CRITICAL)

These functions are essential for UVM's resource database, factory, and component hierarchy matching.

| Function | Signature | C Implementation | Usage in UVM |
|----------|-----------|------------------|--------------|
| `uvm_re_match` | `int(string re, string str, bit deglob)` | POSIX regcomp/regexec | Used by `uvm_is_match()` in ~20 places |
| `uvm_glob_to_re` | `string(string glob)` | Custom glob-to-regex | Resource scope conversion |
| `uvm_re_deglobbed` | `string(string glob, bit with_brackets)` | Custom | Convert glob to POSIX regex |
| `uvm_re_buffer` | `string()` | Static buffer | Error message storage |
| `uvm_re_comp` | `chandle(string re, bit deglob)` | regcomp | Compile regex |
| `uvm_re_exec` | `int(chandle rexp, string str)` | regexec | Execute compiled regex |
| `uvm_re_free` | `void(chandle rexp)` | regfree | Free compiled regex |
| `uvm_re_compexec` | `chandle(string re, string str, bit deglob, output int exec_ret)` | Combined | Compile+execute |
| `uvm_re_compexecfree` | `bit(string re, string str, bit deglob, output int exec_ret)` | Combined | Compile+execute+free |

**No-DPI Fallback**: UVM provides a pure SystemVerilog glob matcher in `uvm_regex.svh` (lines 100-149) that handles basic `*` and `?` wildcards but NOT full regex patterns.

**Recommendation**:
- Short term: Compile UVM with `+define+UVM_REGEX_NO_DPI` to use SV-only glob matching
- Long term: Implement regex runtime functions linking to POSIX regex or RE2

### 2. HDL Backdoor Access (Required for RAL)

Used by UVM's Register Abstraction Layer (RAL) for backdoor read/write access.

| Function | Signature | C Implementation | Usage |
|----------|-----------|------------------|-------|
| `uvm_hdl_check_path` | `int(string path)` | VPI vpi_handle_by_name | Check if HDL path exists |
| `uvm_hdl_read` | `int(string path, output uvm_hdl_data_t value)` | VPI vpi_get_value | Read HDL signal |
| `uvm_hdl_deposit` | `int(string path, uvm_hdl_data_t value)` | VPI vpi_put_value | Write HDL signal |
| `uvm_hdl_force` | `int(string path, uvm_hdl_data_t value)` | VPI vpi_put_value(vpiForceFlag) | Force HDL signal |
| `uvm_hdl_release` | `int(string path)` | VPI vpi_put_value(vpiReleaseFlag) | Release forced signal |
| `uvm_hdl_release_and_read` | `int(string path, inout uvm_hdl_data_t value)` | Combined | Release and read |

**No-DPI Fallback**: Functions call `uvm_report_fatal()` - no fallback implementation.

**Recommendation**:
- These require VPI which implies a full simulator
- For CIRCT, implement stub functions that return failure (0) or success (1) based on use case
- Actual implementation would need CIRCT's own hierarchical name resolution

### 3. Command-line Processing (Required)

Used for `+UVM_*` plusarg parsing.

| Function | Signature | C Implementation | Usage |
|----------|-----------|------------------|-------|
| `uvm_dpi_get_next_arg_c` | `string(int init)` | VPI vpi_get_vlog_info | Iterate command-line args |
| `uvm_dpi_get_tool_name_c` | `string()` | VPI info.product | Get simulator name |
| `uvm_dpi_get_tool_version_c` | `string()` | VPI info.version | Get simulator version |

**No-DPI Fallback**: Return empty string, "?", "?" respectively.

**Recommendation**:
- Implement `uvm_dpi_get_next_arg_c` to return CIRCT runtime arguments
- `uvm_dpi_get_tool_name_c` can return "CIRCT"
- `uvm_dpi_get_tool_version_c` can return CIRCT version

### 4. Polling/Callbacks (Optional)

For advanced signal change monitoring. Only enabled with `+define+UVM_PLI_POLLING_ENABLE`.

| Function | Signature | Usage |
|----------|-----------|-------|
| `uvm_polling_create` | `chandle(string name, int sv_key)` | Create polling handle |
| `uvm_polling_set_enable_callback` | `void(chandle hnd, int enable)` | Enable/disable callbacks |
| `uvm_polling_get_callback_enable` | `int(chandle hnd)` | Query callback state |
| `uvm_polling_setup_notifier` | `int(string fullname)` | Setup signal notifier |
| `uvm_polling_process_changelist` | `void()` | Process pending changes |
| `uvm_hdl_signal_size` | `int(string path)` | Get signal width |

**Recommendation**: Not critical for initial UVM support. Can be stubbed.

## Implementation Strategy

### Phase 1: Compile-time Workaround (Immediate)

Compile UVM with no-DPI defines:
```bash
+define+UVM_NO_DPI
```

This enables:
- SV-only glob matching (basic `*` and `?` patterns)
- Empty command-line args
- Fatal errors on HDL backdoor access (acceptable if not using RAL backdoor)

### Phase 2: Essential Runtime Functions (Short-term)

Implement these 6 functions in CIRCT's runtime library:

1. **`__uvm_dpi_get_tool_name`** - Return "CIRCT"
2. **`__uvm_dpi_get_tool_version`** - Return CIRCT version
3. **`__uvm_dpi_get_next_arg`** - Return command-line arguments
4. **`__uvm_re_match_glob`** - Simple glob matching (already in SV)
5. **`__uvm_hdl_check_path`** - Return 0 (path not found)
6. **`__uvm_hdl_read`** - Return 0 (read failed)

### Phase 3: Full Regex Support (Medium-term)

Link CIRCT runtime to POSIX regex or RE2 library:
- Implement `uvm_re_comp`, `uvm_re_exec`, `uvm_re_free`
- Handle `chandle` as opaque pointer to `regex_t`

### Phase 4: HDL Access (Long-term)

Requires CIRCT-internal hierarchical name resolution:
- Track all signals with hierarchical names
- Implement `uvm_hdl_check_path` using internal lookup
- Implement `uvm_hdl_read`/`uvm_hdl_deposit` for register access

## Files Modified for DPI Support

Current implementation in CIRCT:

| File | Purpose |
|------|---------|
| `lib/Conversion/ImportVerilog/Structure.cpp` | DPI function declaration (func.func private) |
| `lib/Conversion/ImportVerilog/Expressions.cpp` | DPI call stub (return default values) |
| `test/Conversion/ImportVerilog/dpi.sv` | Test for DPI stub handling |

## Actual DPI Calls in UVM

When parsing UVM with CIRCT (without UVM_NO_DPI), 24 DPI function calls are encountered:

### Regex Functions (7 calls)
- `uvm_re_compexecfree` - Main regex match function (1 call)
- `uvm_re_buffer` - Error buffer access (1 call)
- `uvm_re_deglobbed` - Glob to regex conversion (1 call)
- `uvm_re_comp` - Compile regex (1 call)
- `uvm_re_exec` - Execute regex (1 call)
- `uvm_re_free` - Free compiled regex (2 calls)

### Command-line Functions (3 calls)
- `uvm_dpi_get_next_arg_c` - Iterate plusargs (1 call)
- `uvm_dpi_get_tool_name_c` - Get simulator name (1 call)
- `uvm_dpi_get_tool_version_c` - Get simulator version (1 call)

### HDL Backdoor Functions (14 calls)
- `uvm_hdl_check_path` - Check path exists (2 calls in uvm_reg_mem_hdl_paths_seq)
- `uvm_hdl_read` - Read HDL value (5 calls in uvm_reg/uvm_mem)
- `uvm_hdl_deposit` - Write HDL value (5 calls in uvm_reg/uvm_mem)
- `uvm_hdl_force` - Force HDL value (1 call)
- `uvm_hdl_release_and_read` - Release and read (1 call)

### Location Summary
| File | DPI Calls |
|------|-----------|
| `uvm_regex.svh` | 3 (uvm_re_*) |
| `uvm_svcmd_dpi.svh` | 6 (uvm_dpi_*, uvm_re_*) |
| `uvm_regex_cache.svh` | 1 (uvm_re_free) |
| `uvm_hdl.svh` | 4 (uvm_hdl_*) |
| `uvm_reg.svh` | 4 (uvm_hdl_read/deposit) |
| `uvm_mem.svh` | 4 (uvm_hdl_read/deposit) |
| `uvm_reg_mem_hdl_paths_seq.svh` | 3 (uvm_hdl_check_path/read) |

## Testing

Run UVM with current CIRCT (DPI calls return stub values):
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv \
    -I ~/uvm-core/src 2>&1 | grep "DPI-C imports"
```

To see all 24 DPI-related remarks:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv \
    -I ~/uvm-core/src 2>&1 | grep -c "DPI-C imports not yet supported"
# Output: 24
```

## References

- UVM 1800.2-2020 Standard, Section 19.6 (HDL Access)
- SystemVerilog IEEE 1800-2017, Chapter 35 (DPI)
- CIRCT Moore Dialect Documentation
- UVM-Core DPI source: `~/uvm-core/src/dpi/`
