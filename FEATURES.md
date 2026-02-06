# circt-sim UVM Simulation Feature Status

This document tracks the feature parity status of `circt-sim` for running
UVM testbenches, compared to commercial simulators like Cadence Xcelium.

## Feature Gap Table

| Feature | Status | Notes |
|---------|--------|-------|
| **UVM Core** | | |
| UVM init / phases | WORKS | `uvm_root`, build/connect/run phases |
| UVM factory | WORKS | `create_object_by_type`, class registration |
| UVM report server | WORKS | Severity actions, report messages |
| `$cast` / RTTI | WORKS | Parent table, type hierarchy checking |
| VTable dispatch | WORKS | Inherited methods, virtual calls across class hierarchy |
| `process::self()` | WORKS | Intercepted for both old and new compilations |
| **Data Structures** | | |
| Associative arrays | WORKS | Auto-create on null, integer and string keys |
| Queues | WORKS | `push_back`, `pop_front`, `size` |
| Dynamic arrays | WORKS | Basic operations |
| Mailboxes | WORKS | DPI-based blocking/non-blocking |
| **Memory / I/O** | | |
| `$readmemh` | WORKS | File I/O for memory initialization |
| `$fopen` / `$fwrite` | WORKS | Basic file operations |
| Malloc / free | WORKS | Heap allocation with unaligned struct layout |
| **Simulation Infrastructure** | | |
| Combined HdlTop+HvlTop | WORKS | Multi-top simulation with BFMs |
| Shutdown cleanup | WORKS | `_exit(0)` skips expensive destructors for large designs |
| Signal driving (`llhd.drv`) | WORKS | Blocking assignments via epsilon delay |
| Signal probing (`llhd.prb`) | WORKS | Read signal values |
| **Known Gaps** | | |
| `+UVM_TESTNAME` plusarg | PARTIAL | Runtime parsing works (CIRCT_UVM_ARGS env var); `$test$plusargs`/`$value$plusargs` stubbed to 0 in ImportVerilog |
| `$urandom` / `$random` | WORKS | mt19937-based RNG via `__moore_urandom()` interceptors |
| `randomize()` (basic) | WORKS | `__moore_randomize_basic()` fills object memory with random bytes |
| `randomize()` (with dist) | WORKS | `__moore_randomize_with_dist()` implements weighted distribution constraints |
| `randomize()` (with ranges) | MISSING | `__moore_randomize_with_ranges()` not yet intercepted |
| Interface ports | MISSING | Required by AXI-VIP and similar verification IPs |
| ClockVar support | MISSING | Needed by some testbenches |
| `%c` format specifier | MISSING | String formatting gap |
| Coverage collection | MISSING | Functional and code coverage not implemented |
| SystemVerilog Assertions (SVA) | MISSING | Runtime assertion checking |
| `$finish` exit code | PARTIAL | Process exits 0 even on `UVM_FATAL` |
| DPI-C imports | PARTIAL | Some intercepted, most stubbed |
| Semaphores | UNKNOWN | May work via DPI stubs |
| Named events | PARTIAL | Basic `wait` / `trigger` works |
| String methods | PARTIAL | Some native implementations, gaps remain |
| Simulation performance | SLOW | Large UVM designs (APB AVIP) take >300s wall-clock |

## AVIP Simulation Status

| AVIP | HvlTop | HdlTop | Combined | Notes |
|------|--------|--------|----------|-------|
| APB | Runs (slow) | Runs | Runs | >300s wall-clock, simulation logic completes |
| AHB | Runs (slow) | Runs | Runs | Similar to APB |
| UART | Runs | Runs | Runs | Reaches ~559.7 us sim time |
| I2S | Runs | - | - | `I2sBaseTest` starts, slow |
| I3C | Error | - | - | `ase_test` not registered (test config, not sim bug) |

## Test Counts

| Suite | Total | Pass | XFail | Unsupported |
|-------|-------|------|-------|-------------|
| circt-sim | 138 | 137 | 1 | 0 |
| sv-tests BMC | 26 | 23 | 3 | 0 |
| sv-tests LEC | 23 | 23 | 0 | 0 |

## Key Fixes History

| Fix | Description |
|-----|-------------|
| hw.array indexing | Element 0 at LSB, `idx * elementWidth` |
| Assoc array auto-create | Create on first access when pointer is null |
| findMemoryBlock for func args | Address-based lookup for function argument refs |
| Assoc array keySize | Truncation fix: `(width+7)/8` |
| Store/Load address fallback | Fall through to `findMemoryBlockByAddress` |
| Static vtable fallback | Eliminates dispatch warnings for corrupt pointers |
| Native store OOB fallback | Falls through when native block too small |
| Shutdown `_exit` | Placed in `processInput()` before destructor runs |
| `randomize_basic` | Fills object memory with random bytes instead of no-op |
| `randomize_with_dist` | Weighted distribution constraints with range/weight arrays |
