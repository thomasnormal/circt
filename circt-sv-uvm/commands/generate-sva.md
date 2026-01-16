---
name: generate-sva
description: Generate SystemVerilog Assertions based on design intent or protocol
argument-hint: "<description_or_protocol>"
allowed-tools:
  - Write
  - Read
  - Glob
---

# SVA Generator Command

Generate SystemVerilog Assertions based on design intent, protocol specifications, or natural language descriptions.

## Instructions

1. **Understand the assertion requirement**:
   - Protocol property (handshake, FIFO, arbiter)
   - Timing relationship (setup/hold, latency)
   - Safety property (mutex, one-hot, valid ranges)
   - Liveness property (eventual response, fairness)

2. **Gather context**:
   - Signal names and types
   - Clock and reset signals
   - Expected timing (cycles, ranges)
   - Edge conditions (posedge, negedge)

3. **Generate appropriate assertion type**:
   - `assert property` for checks
   - `assume property` for constraints
   - `cover property` for coverage

4. **Common assertion patterns**:

### Handshake (req/ack)
```systemverilog
property req_ack_handshake;
  @(posedge clk) disable iff (!rst_n)
  req |-> ##[1:MAX_LATENCY] ack;
endproperty
assert property (req_ack_handshake);
```

### Valid-Ready Protocol
```systemverilog
property valid_stable_until_ready;
  @(posedge clk) disable iff (!rst_n)
  (valid && !ready) |=> valid;
endproperty

property data_stable_when_valid;
  @(posedge clk) disable iff (!rst_n)
  (valid && !ready) |=> $stable(data);
endproperty
```

### One-Hot Encoding
```systemverilog
property one_hot_state;
  @(posedge clk) disable iff (!rst_n)
  $onehot(state);
endproperty
```

### FIFO Properties
```systemverilog
property no_overflow;
  @(posedge clk) disable iff (!rst_n)
  (full && push && !pop) |-> ##1 overflow_error;
endproperty

property no_underflow;
  @(posedge clk) disable iff (!rst_n)
  (empty && pop && !push) |-> ##1 underflow_error;
endproperty
```

### Latency Bounds
```systemverilog
property response_latency;
  @(posedge clk) disable iff (!rst_n)
  request |-> ##[MIN:MAX] response;
endproperty
```

5. **Include supporting constructs**:
   - Sequences for complex patterns
   - Local variables for value tracking
   - Helper functions if needed

## Example Usage

```
/generate-sva valid-ready handshake for AXI
/generate-sva request must be acknowledged within 10 cycles
/generate-sva FIFO never overflows or underflows
/generate-sva state machine is always one-hot
```

## Output Format

Generate complete assertion code with:
- Property definition
- Assert/assume/cover directive
- Optional: bind statement for external use
- Comments explaining the assertion
