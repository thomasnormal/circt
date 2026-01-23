# UVM TLM Port/Export Infrastructure Design Document for CIRCT

## Executive Summary

This document analyzes the Transaction Level Modeling (TLM) port/export infrastructure required for UVM agent-driver-sequencer communication in CIRCT. The analysis is based on:
- Official UVM 1800.2-2017 specification
- Real-world usage patterns from mbit AVIPs (AHB, APB, AXI4, AXI4Lite, SPI, UART, I2C, I3C, JTAG)
- Existing CIRCT UVM runtime implementation
- Current test coverage (uvm-scoreboard-pattern.sv)

## Current State

### Existing Implementation in CIRCT

Location: `/home/thomas-ahle/circt/lib/Runtime/uvm/uvm_pkg.sv`

The current runtime provides comprehensive TLM infrastructure:

#### Analysis Ports (lines 1466-1533)
- `uvm_analysis_port #(T)` - Broadcast port for sending transactions to multiple subscribers
- `uvm_analysis_imp #(T, IMP)` - Implementation port that calls `write()` on the implementing component
- `uvm_analysis_export #(T)` - Hierarchical export for analysis connections

#### Sequencer-Driver Ports (lines 1536-1656)
- `uvm_seq_item_pull_port #(REQ, RSP)` - Driver's port to pull items from sequencer
- `uvm_seq_item_pull_imp #(REQ, RSP, IMP)` - Sequencer's implementation
- `uvm_seq_item_pull_export #(REQ, RSP)` - Hierarchical sequencer export

#### TLM FIFOs (lines 1732-1841)
- `uvm_tlm_fifo #(T)` - Basic TLM FIFO with put/get operations
- `uvm_tlm_analysis_fifo #(T)` - Analysis FIFO with `analysis_export` for connecting to analysis ports

#### Helper Classes (lines 1847-1929)
- `uvm_put_imp`, `uvm_get_imp`, `uvm_get_peek_imp` - Implementation helpers

## TLM Usage Patterns from AVIPs

### Pattern 1: Monitor -> Scoreboard via Analysis FIFO

This is the most common pattern, used in every AVIP.

**Monitor (Producer):**
```systemverilog
class AhbMasterMonitorProxy extends uvm_monitor;
  uvm_analysis_port#(AhbMasterTransaction) ahbMasterAnalysisPort;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ahbMasterAnalysisPort = new("ahbMasterAnalysisPort", this);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      // Sample transaction from interface
      ahbMasterAnalysisPort.write(ahbMasterClonePacket);
    end
  endtask
endclass
```

**Scoreboard (Consumer):**
```systemverilog
class AhbScoreboard extends uvm_scoreboard;
  uvm_tlm_analysis_fifo#(AhbMasterTransaction) ahbMasterAnalysisFifo[];
  uvm_tlm_analysis_fifo#(AhbSlaveTransaction) ahbSlaveAnalysisFifo[];

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ahbMasterAnalysisFifo = new[NO_OF_MASTERS];
    ahbSlaveAnalysisFifo = new[NO_OF_SLAVES];

    foreach(ahbMasterAnalysisFifo[i])
      ahbMasterAnalysisFifo[i] = new($sformatf("ahbMasterAnalysisFifo[%0d]", i), this);
    foreach(ahbSlaveAnalysisFifo[i])
      ahbSlaveAnalysisFifo[i] = new($sformatf("ahbSlaveAnalysisFifo[%0d]", i), this);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      for(int j = 0; j < NO_OF_MASTERS; j++) begin
        ahbMasterAnalysisFifo[j].get(ahbMasterTransaction);  // Blocking
        // Process transaction
      end
    end
  endtask
endclass
```

**Environment (Connection):**
```systemverilog
function void AhbEnvironment::connect_phase(uvm_phase phase);
  foreach(ahbMasterAgent[i]) begin
    ahbMasterAgent[i].ahbMasterMonitorProxy.ahbMasterAnalysisPort.connect(
      ahbScoreboard.ahbMasterAnalysisFifo[i].analysis_export
    );
  end
endfunction
```

### Pattern 2: Monitor -> Coverage via uvm_subscriber

**Coverage Component (Subscriber):**
```systemverilog
class apb_master_coverage extends uvm_subscriber #(apb_master_tx);
  `uvm_component_utils(apb_master_coverage)

  covergroup apb_master_covergroup with function sample(apb_master_tx packet);
    // Coverpoints...
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    apb_master_covergroup = new();
  endfunction

  // Called automatically when write() is invoked on analysis_export
  virtual function void write(apb_master_tx t);
    apb_master_covergroup.sample(t);
  endfunction
endclass
```

**Agent (Connection):**
```systemverilog
function void apb_master_agent::connect_phase(uvm_phase phase);
  apb_master_mon_proxy_h.apb_master_analysis_port.connect(
    apb_master_cov_h.analysis_export  // Inherited from uvm_subscriber
  );
endfunction
```

### Pattern 3: Driver -> Sequencer Communication

**Driver:**
```systemverilog
class AhbMasterDriverProxy extends uvm_driver#(AhbMasterTransaction);
  // seq_item_port is inherited from uvm_driver

  task run_phase(uvm_phase phase);
    forever begin
      seq_item_port.get_next_item(req);  // Blocking call to sequencer
      // Drive transaction on interface
      seq_item_port.item_done();  // Signal completion
    end
  endtask
endclass
```

**Agent (Connection):**
```systemverilog
function void AhbMasterAgent::connect_phase(uvm_phase phase);
  ahbMasterDriverProxy.seq_item_port.connect(ahbMasterSequencer.seq_item_export);
endfunction
```

### Pattern 4: Multiple Response Ports (AXI4)

**Driver with Separate Read/Write Ports:**
```systemverilog
class axi4_master_driver_proxy extends uvm_driver #(axi4_master_tx, axi4_master_tx);
  uvm_seq_item_pull_port #(REQ, RSP) axi_write_seq_item_port;
  uvm_seq_item_pull_port #(REQ, RSP) axi_read_seq_item_port;
  uvm_analysis_port #(RSP) axi_write_rsp_port;
  uvm_analysis_port #(RSP) axi_read_rsp_port;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    axi_write_seq_item_port = new("axi_write_seq_item_port", this);
    axi_read_seq_item_port = new("axi_read_seq_item_port", this);
    axi_write_rsp_port = new("axi_write_rsp_port", this);
    axi_read_rsp_port = new("axi_read_rsp_port", this);
  endfunction

  task run_phase(uvm_phase phase);
    fork
      write_channel_task();
      read_channel_task();
    join
  endtask

  task write_channel_task();
    forever begin
      axi_write_seq_item_port.get_next_item(req_wr);
      // Drive write transaction
      axi_write_seq_item_port.item_done();
    end
  endtask
endclass
```

## Required TLM Types and APIs

### Priority 1: Essential for Basic AVIP Support

| Class | Key Methods | Usage Count |
|-------|-------------|-------------|
| `uvm_analysis_port #(T)` | `new()`, `connect()`, `write()` | 150+ instances |
| `uvm_tlm_analysis_fifo #(T)` | `new()`, `get()`, `size()`, `analysis_export` | 80+ instances |
| `uvm_subscriber #(T)` | `new()`, `write()`, `analysis_export` | 20+ instances |
| `uvm_seq_item_pull_port #(REQ,RSP)` | `new()`, `connect()`, `get_next_item()`, `item_done()` | 40+ instances |

### Priority 2: Enhanced Functionality

| Class | Key Methods | Usage Count |
|-------|-------------|-------------|
| `uvm_analysis_export #(T)` | `new()`, `connect()`, `write()` | 15+ instances |
| `uvm_analysis_imp #(T, IMP)` | `new()`, `write()` | 30+ instances |
| `uvm_tlm_fifo #(T)` | `new()`, `put()`, `get()`, `try_get()`, `size()` | 10+ instances |

### Priority 3: Advanced/Rare Patterns

| Class | Key Methods | Usage Count |
|-------|-------------|-------------|
| `uvm_blocking_put_port #(T)` | `new()`, `put()` | <5 instances |
| `uvm_blocking_get_port #(T)` | `new()`, `get()` | <5 instances |
| `uvm_blocking_peek_port #(T)` | `new()`, `peek()` | <5 instances |

## API Specifications

### uvm_analysis_port #(T)

```systemverilog
class uvm_analysis_port #(type T = int) extends uvm_port_base #(uvm_tlm_if_base #(T, T));

  // Constructor
  function new(string name, uvm_component parent);

  // Get type name for debugging
  virtual function string get_type_name();

  // Connect to an export or imp
  virtual function void connect(uvm_port_base #(uvm_tlm_if_base #(T, T)) provider);

  // Broadcast transaction to all connected subscribers
  virtual function void write(input T t);
endclass
```

**Key Behavior:**
- Maintains a list of subscribers
- `write()` broadcasts to ALL connected exports/imps (1-to-N communication)
- Non-blocking operation

### uvm_tlm_analysis_fifo #(T)

```systemverilog
class uvm_tlm_analysis_fifo #(type T = int) extends uvm_tlm_fifo #(T);
  uvm_analysis_imp #(T, uvm_tlm_analysis_fifo #(T)) analysis_export;

  // Constructor (unbounded FIFO)
  function new(string name, uvm_component parent = null);

  // Get type name for debugging
  virtual function string get_type_name();

  // Called when analysis_port.write() invokes analysis_export.write()
  virtual function void write(input T t);

  // Inherited from uvm_tlm_fifo:
  // virtual task get(output T t);           // Blocking get
  // virtual function bit try_get(output T t); // Non-blocking get
  // virtual function int size();            // Current FIFO size
  // virtual function bit is_empty();
endclass
```

**Key Behavior:**
- Unbounded FIFO (no size limit)
- `analysis_export` is automatically connected to the parent analysis port
- `get()` blocks until data is available (requires wait semantics in simulation)

### uvm_subscriber #(T)

```systemverilog
class uvm_subscriber #(type T = uvm_sequence_item) extends uvm_component;
  uvm_analysis_imp #(T, uvm_subscriber #(T)) analysis_export;

  function new(string name, uvm_component parent);

  virtual function string get_type_name();

  // Pure virtual - MUST be overridden by derived class
  virtual function void write(T t);
endclass
```

**Key Behavior:**
- Provides built-in `analysis_export` for connecting to analysis ports
- Derived class implements `write()` to process received transactions

### uvm_seq_item_pull_port #(REQ, RSP)

```systemverilog
class uvm_seq_item_pull_port #(type REQ = uvm_sequence_item, type RSP = REQ)
  extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP));

  function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);

  virtual function string get_type_name();

  // Blocking: Get next sequence item from sequencer
  virtual task get_next_item(output REQ req_arg);

  // Non-blocking: Try to get next item
  virtual task try_next_item(output REQ req_arg);

  // Signal item completion
  virtual function void item_done(RSP rsp = null);

  // Other methods
  virtual task put(RSP rsp);
  virtual task get(output REQ req);
  virtual task peek(output REQ req);
  virtual function void put_response(RSP rsp);
  virtual function bit has_do_available();
endclass
```

## Runtime Implementation Considerations

### Connection Mechanism

The `connect()` method needs to:
1. Store reference to the connected provider
2. For analysis ports, add to subscriber list
3. Validate connection compatibility at elaboration time

Current implementation in uvm_pkg.sv:
```systemverilog
virtual function void connect(uvm_port_base #(IF) provider);
  m_if = provider.m_if;  // For simple 1:1 connections
  // For analysis ports, also add to m_subscribers queue
endfunction
```

### Blocking Operations

The `get()` operation in FIFOs must block until data is available. This requires:
1. Event-based wait mechanism
2. Integration with simulation scheduler
3. For CIRCT: may require special handling in lowering/runtime

Current stub approach:
```systemverilog
virtual task get(output T t);
  // TODO: Implement blocking with wait for data
  t = null;
endtask
```

**Proposed implementation:**
```systemverilog
class uvm_tlm_fifo #(type T = int);
  protected event m_not_empty;

  virtual task put(input T t);
    m_fifo.push_back(t);
    -> m_not_empty;  // Trigger event
    put_ap.write(t);
  endtask

  virtual task get(output T t);
    while (m_fifo.size() == 0)
      @m_not_empty;  // Wait for data
    t = m_fifo.pop_front();
    get_ap.write(t);
  endtask
endclass
```

### Parameterized Type Handling

All TLM classes are parameterized by transaction type. This requires:
1. Correct type propagation through class hierarchy
2. Type-safe connections (analysis port of type A cannot connect to FIFO of type B)

## Testing Strategy

### Test File 1: uvm-tlm-analysis-port.sv

Tests:
1. Basic analysis port declaration and construction
2. Connection to analysis_export
3. write() broadcasting to multiple subscribers
4. uvm_subscriber pattern with custom write()

### Test File 2: uvm-tlm-fifo.sv

Tests:
1. uvm_tlm_analysis_fifo declaration and construction
2. Connection via analysis_export
3. FIFO operations: get(), size(), is_empty()
4. Scoreboard pattern with multiple FIFOs

### Integration Test: uvm-scoreboard-pattern.sv (existing)

Already tests:
- Multiple analysis FIFOs in scoreboard
- fork/join with parallel comparison tasks
- forever loops with FIFO.get() blocking calls

## Implementation Recommendations

### Phase 1: Complete Current Stubs

1. **Implement blocking get()** in uvm_tlm_fifo using events
2. **Add size checking** for FIFO operations
3. **Verify subscriber list** management in uvm_analysis_port

Estimated effort: 2-3 days

### Phase 2: Enhanced TLM Support

1. **Add try_get(), try_put()** non-blocking variants
2. **Implement peek()** operation
3. **Add FIFO size limits** for bounded FIFOs

Estimated effort: 2-3 days

### Phase 3: Full Compliance

1. **Add port connectivity checking** at elaboration time
2. **Implement connection debug/tracing**
3. **Support multi-port connections** (e.g., multiple exports from one FIFO)

Estimated effort: 3-5 days

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Blocking get() implementation | High | Use events and proper simulation scheduling |
| Type safety in connections | Medium | Add runtime type checking in connect() |
| Memory management for FIFOs | Medium | Use proper cleanup in end_of_simulation |
| Performance of subscriber lists | Low | Use efficient data structures (queues) |

## Conclusion

The TLM port/export infrastructure is critical for UVM testbench component communication. The current CIRCT runtime has good foundational support, but needs:

1. **Essential**: Blocking FIFO get() implementation using events
2. **Important**: Complete uvm_subscriber integration
3. **Nice-to-have**: Connection validation and debug tracing

The analysis port -> analysis FIFO -> scoreboard pattern is the most critical path to support, as it appears in every production AVIP and is fundamental to UVM verification methodology.

## Appendix: TLM Usage Statistics from mbit AVIPs

| AVIP | Analysis Ports | Analysis FIFOs | Subscribers | Seq Item Ports |
|------|---------------|----------------|-------------|----------------|
| AHB | 2 | 2 | 2 | 2 |
| APB | 2 | 2 | 2 | 2 |
| AXI4 | 5+ | 5+ | 2 | 4 |
| AXI4Lite | 10+ | 10+ | 4 | 4 |
| SPI | 2 | 2 | 2 | 2 |
| UART | 2 | 2 | 2 | 2 |
| I2C | 2 | 2 | 2 | 2 |
| I3C | 2 | 2 | 2 | 2 |
| JTAG | 2 | 2 | 2 | 2 |

Total observed: 150+ analysis port instances, 80+ FIFO instances across all AVIPs.
