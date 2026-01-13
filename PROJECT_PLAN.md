# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.

## Current Status: UVM Parsing - 0 ERRORS, 7 WARNINGS

### Recent Commits (40+ this session)
- `fe922d3c9` - queue delete(index), triggered, unique, wait
- `da49e34b3` - %p format specifier
- `a7a396879` - string character assignment
- `4a039b04d` - SystemVerilog interface support
- `e56e0f30c` - Moore runtime library
- `7a5e7d892` - MooreToCore queue/array lowering
- `4b32f532d` - static class properties
- `6674791b7` - associative array iterator methods
- `f6bb2c6a9` - dynamic array new[size]
- `45ffe6838` - fork/join, named_block, disable operations
- And many more...

## Remaining UVM Warnings (Informational)
1. `no top-level modules found` - Expected, UVM is a package
2. `unknown escape sequence '\.'` - Regex in string literal
3. `streaming concatenation not fully supported` - Partial support
4. `static class property could not be resolved` - Static resolution limitation

## Features Completed

### Class Support
- [x] Class declarations and handles
- [x] Class inheritance (extends)
- [x] Virtual methods and vtables
- [x] Static class properties (partial)
- [x] Parameterized classes
- [x] this_type pattern
- [x] $cast dynamic type checking
- [x] Class handle comparison (==, !=, null)
- [x] new() allocation

### Queue/Array Support
- [x] Queue type and operations
- [x] push_back, push_front, pop_back, pop_front
- [x] delete(), delete(index)
- [x] size(), max(), min(), unique()
- [x] sort()
- [x] Dynamic arrays with new[size]
- [x] Associative arrays
- [x] exists(), delete(key)
- [x] first(), next(), last(), prev() iterators

### String Support
- [x] String type
- [x] itoa(), len(), getc()
- [x] toupper(), tolower()
- [x] putc() character assignment
- [x] %p format specifier

### Process Control
- [x] fork/join, fork/join_any, fork/join_none
- [x] Named blocks
- [x] disable statement
- [x] wait(condition) statement

### Event Support
- [x] event type (now using moore::EventType)
- [x] .triggered property
- [x] Event trigger (->)

### Interface Support
- [x] Interface declarations
- [x] Modports
- [x] Virtual interfaces (basic)

### Runtime Library
- [x] Queue operations (__moore_queue_*)
- [x] Dynamic array operations
- [x] String operations
- [x] Built and tested

## Known Limitations / TODO

### High Priority (Blocking AVIP compilation)
1. **UVM Macro Expansion** - AVIPs use UVM macros (`uvm_component_utils, etc.)
2. **Interface Hierarchical References** - `intf.signal` access patterns
3. **Complete Compilation Flow** - Multi-file compilation with dependencies

### Medium Priority
4. **Streaming Concatenation** - Full support for queues/dynamic arrays
5. **Static Class Property Resolution** - Global variable linkage
6. **Coverage Groups** - `covergroup`, `coverpoint`, `cross`
7. **Constraints** - More constraint types
8. **Randomization** - `randomize()` with constraints

### Lower Priority
9. **Assertions** - More assertion types (SVA)
10. **DPI** - SystemVerilog DPI imports
11. **Bind** - Module binding
12. **Program Blocks** - `program` support

## Track Status

### Track A: Process Control (COMPLETE)
- [x] fork/join operations
- [x] Named blocks
- [x] disable statement
- [x] Merged to main

### Track B: Class Infrastructure (COMPLETE)
- [x] Inheritance and vtables
- [x] Static properties
- [x] $cast support
- [x] Merged to main

### Track C: Coverage (PARTIAL)
- [x] Basic coverage ops
- [ ] covergroup implementation
- [ ] Cross coverage

### Track D: LSP Support (DEFERRED)
- [ ] go-to-definition
- [ ] API compatibility issues with slang

## AVIP Testing Status

AVIPs in ~/mbit/* require:
1. UVM package (now 0 errors)
2. Globals packages per AVIP
3. HDL interfaces
4. Proper include paths

Current blockers:
- Missing interface modules
- Missing package imports
- Macro expansion not supported

## Next Steps

1. **Interface Hierarchical Access** - Enable `intf.signal` patterns
2. **Multi-file Compilation** - Proper dependency resolution
3. **UVM Macros** - Either preprocess or implement macro expansion
4. **MooreToCore Completion** - Lower all Moore ops to executable IR
5. **circt-sim Integration** - Run lowered testbenches

## Build Commands
```bash
# Build
ninja -C build circt-verilog

# Test UVM
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src

# Test AVIP (example)
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv \
  -I ~/uvm-core/src \
  ~/mbit/spi_avip/src/globals/SpiGlobalsPkg.sv \
  ~/mbit/spi_avip/src/hdlTop/spiInterface/SpiInterface.sv \
  -I ~/mbit/spi_avip/src
```
