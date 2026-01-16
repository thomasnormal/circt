---
name: UVM Methodology
description: This skill should be used when the user asks about "UVM testbench", "UVM agent", "UVM sequence", "uvm_component", "UVM factory", "UVM phases", "UVM configuration", "UVM RAL", "register abstraction layer", "UVM coverage", "uvm_driver", "uvm_monitor", "uvm_scoreboard", "uvm_env", "uvm_test", "sequence_item", "uvm_sequence", "TLM ports", "analysis ports", "uvm_config_db", or any UVM verification methodology questions.
version: 1.0.0
---

# UVM Methodology Guide

Apply Universal Verification Methodology best practices when helping with UVM testbenches.

## Testbench Architecture

A standard UVM testbench follows this hierarchy:

```
uvm_test
└── uvm_env
    ├── uvm_agent (active/passive)
    │   ├── uvm_sequencer
    │   ├── uvm_driver
    │   └── uvm_monitor
    ├── uvm_scoreboard
    ├── uvm_subscriber (coverage)
    └── virtual_sequencer
```

## Component Implementation Patterns

### Base Class Registration
All UVM components must register with the factory:

```systemverilog
class my_agent extends uvm_agent;
  `uvm_component_utils(my_agent)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
endclass
```

For objects (transactions, sequences):
```systemverilog
class my_transaction extends uvm_sequence_item;
  `uvm_object_utils(my_transaction)

  function new(string name = "my_transaction");
    super.new(name);
  endfunction
endclass
```

### Configuration Pattern
Use `uvm_config_db` to pass configuration:

```systemverilog
// In test or env - set config
uvm_config_db#(my_config)::set(this, "env.agent*", "config", cfg);
uvm_config_db#(virtual my_if)::set(this, "env.agent*", "vif", vif);

// In component - get config
if (!uvm_config_db#(my_config)::get(this, "", "config", cfg))
  `uvm_fatal("CONFIG", "Failed to get config")
```

### Agent Structure
```systemverilog
class my_agent extends uvm_agent;
  my_driver    drv;
  my_monitor   mon;
  my_sequencer sqr;

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    mon = my_monitor::type_id::create("mon", this);
    if (get_is_active() == UVM_ACTIVE) begin
      drv = my_driver::type_id::create("drv", this);
      sqr = my_sequencer::type_id::create("sqr", this);
    end
  endfunction

  function void connect_phase(uvm_phase phase);
    if (get_is_active() == UVM_ACTIVE)
      drv.seq_item_port.connect(sqr.seq_item_export);
  endfunction
endclass
```

### Driver Pattern
```systemverilog
class my_driver extends uvm_driver #(my_transaction);
  virtual my_if vif;

  task run_phase(uvm_phase phase);
    forever begin
      seq_item_port.get_next_item(req);
      drive_transaction(req);
      seq_item_port.item_done();
    end
  endtask

  task drive_transaction(my_transaction tr);
    // Drive signals on vif
  endtask
endclass
```

### Monitor Pattern
```systemverilog
class my_monitor extends uvm_monitor;
  uvm_analysis_port #(my_transaction) ap;
  virtual my_if vif;

  function void build_phase(uvm_phase phase);
    ap = new("ap", this);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      my_transaction tr = my_transaction::type_id::create("tr");
      collect_transaction(tr);
      ap.write(tr);
    end
  endtask
endclass
```

### Sequence Pattern
```systemverilog
class my_sequence extends uvm_sequence #(my_transaction);
  `uvm_object_utils(my_sequence)

  task body();
    repeat (10) begin
      `uvm_do_with(req, { data inside {[0:255]}; })
    end
  endtask
endclass
```

### Scoreboard Pattern
```systemverilog
class my_scoreboard extends uvm_scoreboard;
  uvm_analysis_imp #(my_transaction, my_scoreboard) expected_imp;
  uvm_analysis_imp #(my_transaction, my_scoreboard) actual_imp;
  my_transaction expected_queue[$];

  function void write_expected(my_transaction tr);
    expected_queue.push_back(tr);
  endfunction

  function void write_actual(my_transaction tr);
    my_transaction exp = expected_queue.pop_front();
    if (!tr.compare(exp))
      `uvm_error("MISMATCH", "Transaction mismatch")
  endfunction
endclass
```

## UVM Phases

Phases execute in this order:

**Build phases** (top-down):
1. `build_phase` - Create components
2. `connect_phase` - Connect TLM ports
3. `end_of_elaboration_phase` - Final configuration

**Run phases** (parallel):
4. `run_phase` - Main test execution

**Cleanup phases** (bottom-up):
5. `extract_phase` - Extract data
6. `check_phase` - Check results
7. `report_phase` - Report results

## Factory Overrides

Override component types for test customization:

```systemverilog
// Type override (affects all instances)
my_driver::type_id::set_type_override(my_error_driver::type_id::get());

// Instance override (specific path)
my_driver::type_id::set_inst_override(my_error_driver::type_id::get(),
                                       "env.agent.drv");
```

## Register Abstraction Layer (RAL)

For register access verification:

```systemverilog
class my_reg_block extends uvm_reg_block;
  rand my_reg STATUS;
  rand my_reg CONTROL;

  function void build();
    STATUS = my_reg::type_id::create("STATUS");
    STATUS.configure(this, null, "");
    STATUS.build();

    default_map = create_map("default_map", 0, 4, UVM_LITTLE_ENDIAN);
    default_map.add_reg(STATUS, 'h0, "RW");
    default_map.add_reg(CONTROL, 'h4, "RW");
  endfunction
endclass

// Usage in sequence
reg_block.STATUS.write(status, 'hFF, .parent(this));
reg_block.CONTROL.read(status, value, .parent(this));
```

## TLM Communication

**Analysis ports** (1-to-many broadcast):
```systemverilog
uvm_analysis_port #(transaction) ap;      // In monitor
uvm_analysis_imp #(transaction, sb) imp;  // In scoreboard
```

**Sequencer-driver** (request-response):
```systemverilog
uvm_seq_item_pull_port #(REQ, RSP) seq_item_port;  // In driver
uvm_seq_item_pull_imp #(REQ, RSP, sequencer) seq_item_export;  // In sequencer
```

## Coverage Integration

```systemverilog
class my_coverage extends uvm_subscriber #(my_transaction);
  covergroup cg;
    cp_data: coverpoint tr.data { bins ranges[] = {[0:63], [64:127], [128:255]}; }
    cp_addr: coverpoint tr.addr;
    cross_da: cross cp_data, cp_addr;
  endgroup

  function void write(my_transaction t);
    tr = t;
    cg.sample();
  endfunction
endclass
```

## Common Macros

- `uvm_component_utils(T)` - Register component with factory
- `uvm_object_utils(T)` - Register object with factory
- `uvm_do(item)` - Create and send sequence item
- `uvm_do_with(item, constraints)` - Create with constraints
- `uvm_info(ID, MSG, VERBOSITY)` - Info message
- `uvm_error(ID, MSG)` - Error message
- `uvm_fatal(ID, MSG)` - Fatal message (ends simulation)
