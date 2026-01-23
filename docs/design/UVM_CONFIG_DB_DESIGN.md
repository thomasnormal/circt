# UVM config_db Design Document for CIRCT

## Executive Summary

This document analyzes the requirements for implementing UVM config_db in CIRCT and provides recommendations for a minimum viable implementation. The analysis is based on:
- Official UVM 1800.2-2017 source code from sv-tests
- Real-world usage patterns from mbit AVIPs (AXI4Lite, UART, SPI, I2C, I3C, JTAG)
- Verilator-verification test suite
- Existing CIRCT UVM runtime stubs

## Implementation Status

**Last Updated:** 2026-01-23

### Phase 1 (Minimum Viable) - COMPLETED

The following features have been implemented in the CIRCT UVM runtime:

| Feature | Status | Location |
|---------|--------|----------|
| `uvm_is_match()` glob matching | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| `uvm_re_match()` compat function | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| `uvm_config_db::set()` with wildcards | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| `uvm_config_db::get()` with wildcard lookup | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| `uvm_config_db::exists()` with wildcard support | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| Hierarchical path construction | Done | `lib/Runtime/uvm/uvm_pkg.sv` |
| Unit tests | Done | `test/Runtime/uvm/config_db_test.sv` |

### Supported Patterns

The implementation now supports:
- `"*"` - Match any path (global wildcard)
- `"*agent*"` - Match paths containing "agent"
- `"env.*"` - Match any child of env
- `"env.agent?"` - Match env.agent with single char suffix
- Combined wildcards like `"*.*agent*"`

### Lookup Priority

1. Exact match on full path
2. Wildcard pattern matches (last matching pattern wins)
3. Fallback lookups for compatibility

## Current State

### Existing Implementation in CIRCT

Location: `/home/thomas-ahle/circt/lib/Runtime/uvm/uvm_pkg.sv`

The implementation provides:
- `set()` - Key-value storage with wildcard pattern support
- `get()` - Hierarchical lookup with wildcard pattern matching
- `exists()` - Key existence check with wildcard support
- `wait_modified()` - Stub (non-blocking)
- `uvm_is_match()` - Full glob-style pattern matching

Remaining limitations:
1. No uvm_resource_db integration (not needed for most AVIP use cases)
2. No build-phase precedence handling (last-set-wins only)
3. `wait_modified()` is a non-blocking stub
4. No tracing support (`+UVM_CONFIG_DB_TRACE`)

## API Requirements

### Core Methods (IEEE 1800.2-2017 C.4.2.2)

```systemverilog
class uvm_config_db #(type T = int);
  // Set a configuration value
  static function void set(
    uvm_component cntxt,    // Context component (null = root)
    string inst_name,       // Instance path (supports wildcards)
    string field_name,      // Field/property name
    T value                 // Value to store
  );

  // Get a configuration value
  static function bit get(
    uvm_component cntxt,    // Context component (null = root)
    string inst_name,       // Instance path
    string field_name,      // Field/property name
    inout T value           // Output value
  );

  // Check if configuration exists
  static function bit exists(
    uvm_component cntxt,
    string inst_name,
    string field_name,
    bit spell_chk = 0       // Optional spell checking
  );

  // Wait for configuration modification (task)
  static task wait_modified(
    uvm_component cntxt,
    string inst_name,
    string field_name
  );
endclass
```

### Predefined Type Aliases

```systemverilog
typedef uvm_config_db#(uvm_bitstream_t) uvm_config_int;
typedef uvm_config_db#(string) uvm_config_string;
typedef uvm_config_db#(uvm_object) uvm_config_object;
typedef uvm_config_db#(uvm_object_wrapper) uvm_config_wrapper;
```

## Data Types Used in Practice

Analysis of 200+ config_db calls across mbit AVIPs reveals:

### Primary Categories

| Category | Count | Examples |
|----------|-------|----------|
| Configuration Objects | 85% | `UartEnvConfig`, `Axi4LiteSlaveWriteAgentConfig` |
| Virtual Interfaces | 12% | `virtual UartTxDriverBfm`, `virtual mem_if` |
| Enums | 2% | `uvm_active_passive_enum` |
| Primitives | 1% | `int`, `bit`, `string` |

### Configuration Object Characteristics

Typical config objects extend `uvm_object` and contain:
- Bits/flags (`hasScoreboard`, `hasVirtualSequencer`)
- Integers (`noOfSlaves`, `ADDRESS_WIDTH`)
- Nested config handles (`uartTxAgentConfig`, `spiSlaveAgentConfig[]`)
- Address ranges (arrays of bit vectors)

## Hierarchical Path Lookup

### Path Construction

The full lookup path is constructed as:
```
if (cntxt == null)
  path = inst_name
else if (cntxt.get_full_name() == "")
  path = inst_name
else
  path = {cntxt.get_full_name(), ".", inst_name}
```

### Wildcard Matching

Real-world patterns observed:
- `"*"` - Match any path
- `"*agent*"` - Match paths containing "agent"
- `$sformatf("Axi4LiteMasterWriteAgentConfig[%0d]", i)` - Indexed names

### Precedence Rules

1. During build phase: Higher hierarchy = higher precedence
2. After build: Last-set wins (default precedence)
3. Explicit precedence can be set via resource pool

## Implementation Recommendations

### Phase 1: Minimum Viable Implementation

Goals: Support the most common usage patterns from AVIPs

1. **Key Construction**
   - Proper hierarchical path construction using component context
   - Format: `{full_path}.{inst_name}.{field_name}`

2. **Wildcard Support**
   - Convert glob patterns to regex for matching
   - Support `*` (any string) and basic patterns

3. **Type Storage**
   - Continue using parameterized class approach
   - Store values in per-type associative arrays

4. **Lookup Priority**
   - Check exact match first
   - Then try wildcard patterns in order of specificity
   - Fall back to field_name only match

Estimated complexity: Medium (2-3 days)

### Phase 2: Enhanced Implementation

1. **Resource Database Integration**
   - Proper uvm_resource_db inheritance
   - Resource pool management

2. **Precedence Handling**
   - Track set-time for precedence ordering
   - Support build vs. runtime precedence rules

3. **wait_modified Support**
   - Event-based notification on value changes
   - Waiter list management

4. **Tracing/Debug**
   - `+UVM_CONFIG_DB_TRACE` support
   - Display read/write operations

Estimated complexity: High (5-7 days)

### Phase 3: Full Compliance

1. **Spell Checking**
   - Fuzzy matching for exists() with spell_chk=1

2. **Resource Type Queries**
   - Support for `get_by_type()`

3. **Complete Compatibility**
   - All edge cases from UVM reference implementation

Estimated complexity: High (3-5 days)

## Minimum Viable Implementation Pseudocode

```cpp
// In MooreRuntime.cpp or separate config_db implementation

namespace {

struct ConfigEntry {
  std::string path;      // Full hierarchical path (may contain wildcards)
  std::string field;     // Field name
  void* value;           // Pointer to value storage
  size_t value_size;     // Size of value
  int precedence;        // For ordering during build phase
  uint64_t set_time;     // Monotonic counter for last-wins ordering
};

// Type-specific storage (one per instantiated parameterized type)
// This is naturally handled by SV parameterized class statics
std::map<std::string, ConfigEntry> config_entries;

bool matches_glob(const std::string& pattern, const std::string& path) {
  // Convert pattern to regex:
  // * -> .*
  // ? -> .
  // . -> \.
  std::regex re = glob_to_regex(pattern);
  return std::regex_match(path, re);
}

} // namespace

// set implementation
extern "C" void __moore_config_db_set(
    void* cntxt_ptr,
    const char* inst_name,
    const char* field_name,
    void* value,
    size_t value_size,
    int type_id) {

  std::string path = construct_path(cntxt_ptr, inst_name);
  std::string key = make_key(path, field_name, type_id);

  auto& entry = config_entries[key];
  entry.path = path;
  entry.field = field_name;
  entry.value = allocate_and_copy(value, value_size);
  entry.value_size = value_size;
  entry.set_time = get_monotonic_counter();
}

// get implementation
extern "C" bool __moore_config_db_get(
    void* cntxt_ptr,
    const char* inst_name,
    const char* field_name,
    void* value_out,
    size_t value_size,
    int type_id) {

  std::string lookup_path = construct_path(cntxt_ptr, inst_name);

  // 1. Try exact match
  std::string exact_key = make_key(lookup_path, field_name, type_id);
  if (config_entries.count(exact_key)) {
    copy_value(config_entries[exact_key].value, value_out, value_size);
    return true;
  }

  // 2. Try wildcard patterns (sorted by specificity, then by set_time)
  std::vector<ConfigEntry*> matches;
  for (auto& [key, entry] : config_entries) {
    if (entry.field == field_name &&
        matches_glob(entry.path, lookup_path)) {
      matches.push_back(&entry);
    }
  }

  if (!matches.empty()) {
    // Sort by precedence (descending), then by set_time (descending)
    std::sort(matches.begin(), matches.end(), ...);
    copy_value(matches[0]->value, value_out, value_size);
    return true;
  }

  return false;
}
```

## Testing Strategy

### Unit Tests

1. Basic set/get with exact paths
2. Wildcard pattern matching
3. Type parameterization (int, string, class objects)
4. Virtual interface storage (handle passing)

### Integration Tests

1. Port existing AVIP test patterns
2. Verify compatibility with mem-tb from verilator-verification
3. Test hierarchical lookup with component tree

### Reference Test Cases

From mbit AVIPs:
```systemverilog
// Test 1: Virtual interface passing
uvm_config_db#(virtual UartTxDriverBfm)::set(null, "*", "uartTxDriverBfm", bfm);
uvm_config_db#(virtual UartTxDriverBfm)::get(this, "", "uartTxDriverBfm", bfm);

// Test 2: Config object passing
uvm_config_db#(UartEnvConfig)::set(this, "*", "uartEnvConfig", cfg);
uvm_config_db#(UartEnvConfig)::get(this, "", "uartEnvConfig", cfg);

// Test 3: Indexed field names
uvm_config_db#(AgentConfig)::set(this, "*", $sformatf("cfg[%0d]", i), cfg[i]);

// Test 4: Wildcard instance paths
uvm_config_db#(AgentConfig)::set(this, "*agent*", "config", cfg);
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Type safety at runtime | Medium | Use type ID tags, clear error messages |
| Glob pattern complexity | Low | Start with simple patterns, expand later |
| Memory management | Medium | Use reference counting for stored objects |
| Component hierarchy timing | High | Ensure get_full_name() works during build |

## Conclusion

A minimum viable config_db implementation requires:

1. **Essential**: Hierarchical path construction, basic wildcard matching, per-type storage
2. **Important**: Virtual interface handle support, precedence ordering
3. **Deferrable**: wait_modified, spell checking, tracing

The primary challenge is not the API itself (which is well-defined) but ensuring seamless integration with CIRCT's lowering pipeline and runtime object model. The existing associative array infrastructure in MooreRuntime.cpp provides a solid foundation for the key-value storage aspects.

Recommended starting point: Enhance the existing uvm_pkg.sv stub with proper path construction and wildcard matching, while keeping the storage mechanism simple. This will immediately enable most AVIP test cases to parse and potentially execute.
