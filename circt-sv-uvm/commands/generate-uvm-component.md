---
name: generate-uvm-component
description: Generate UVM component boilerplate (agent, driver, monitor, sequence, etc.)
argument-hint: "<component_type> <name>"
allowed-tools:
  - Write
  - Read
  - Glob
---

# UVM Component Generator Command

Generate well-structured UVM component boilerplate following best practices.

## Instructions

1. **Parse arguments** to determine:
   - Component type: agent, driver, monitor, sequencer, sequence, scoreboard, env, test, subscriber, coverage
   - Component name (e.g., `apb`, `axi`, `my_protocol`)

2. **Ask for additional context** if needed:
   - Transaction/sequence_item class name
   - Interface name (for drivers/monitors)
   - Parent agent name (for sub-components)

3. **Generate component** following UVM conventions:
   - Proper class hierarchy (`extends uvm_*`)
   - Factory registration (`uvm_component_utils` / `uvm_object_utils`)
   - Standard phase methods
   - TLM ports where appropriate
   - Configuration object integration

4. **Component templates**:

### Agent
- Contains driver, monitor, sequencer
- Active/passive mode support
- Configuration object
- Analysis ports

### Driver
- `run_phase` with `seq_item_port.get_next_item()`
- Virtual interface handle
- Reset handling

### Monitor
- Virtual interface handle
- Analysis port for observed transactions
- Protocol-specific sampling

### Sequence
- `body()` task implementation
- `uvm_do` / `uvm_do_with` usage
- p_sequencer typedef if needed

### Scoreboard
- Analysis FIFOs for expected/actual
- Comparison logic
- Error reporting

### Coverage
- Covergroups for functional coverage
- Sample on transaction events
- Cross coverage where appropriate

## Example Usage

```
/generate-uvm-component agent apb
/generate-uvm-component sequence axi_write
/generate-uvm-component scoreboard my_checker
/generate-uvm-component driver uart
```

## Best Practices Applied

- Factory registration for all components
- Configuration object pattern
- Virtual interface access via config_db
- Proper phase usage
- TLM communication
- Consistent naming conventions
