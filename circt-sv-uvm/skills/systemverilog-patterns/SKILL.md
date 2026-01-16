---
name: SystemVerilog Patterns
description: This skill should be used when the user asks about "SystemVerilog coding", "SV interface", "modport", "clocking block", "SystemVerilog class", "SV constraints", "randomization", "SV assertions", "SVA", "always_ff", "always_comb", "unique case", "priority case", "packed struct", "unpacked array", "dynamic array", "associative array", "queue", "mailbox", "semaphore", "event", or SystemVerilog coding style and best practices.
version: 1.0.0
---

# SystemVerilog Patterns Guide

Apply SystemVerilog best practices for RTL design and verification.

## Interface Patterns

### Basic Interface with Modports
```systemverilog
interface apb_if(input logic clk, rst_n);
  logic        psel;
  logic        penable;
  logic        pwrite;
  logic [31:0] paddr;
  logic [31:0] pwdata;
  logic [31:0] prdata;
  logic        pready;
  logic        pslverr;

  modport master(
    output psel, penable, pwrite, paddr, pwdata,
    input  prdata, pready, pslverr
  );

  modport slave(
    input  psel, penable, pwrite, paddr, pwdata,
    output prdata, pready, pslverr
  );

  modport monitor(
    input psel, penable, pwrite, paddr, pwdata, prdata, pready, pslverr
  );
endinterface
```

### Clocking Blocks
```systemverilog
interface my_if(input logic clk);
  logic valid, ready;
  logic [7:0] data;

  clocking driver_cb @(posedge clk);
    output valid, data;
    input  ready;
  endclocking

  clocking monitor_cb @(posedge clk);
    input valid, ready, data;
  endclocking

  modport driver(clocking driver_cb);
  modport monitor(clocking monitor_cb);
endinterface
```

## RTL Coding Patterns

### Always Blocks
```systemverilog
// Sequential logic - always_ff
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n)
    state <= IDLE;
  else
    state <= next_state;
end

// Combinational logic - always_comb
always_comb begin
  next_state = state;
  case (state)
    IDLE: if (start) next_state = RUN;
    RUN:  if (done)  next_state = IDLE;
  endcase
end

// Latch inference (when intended) - always_latch
always_latch begin
  if (enable)
    q = d;
end
```

### Case Statement Best Practices
```systemverilog
// unique - exactly one case must match (parallel, full)
always_comb begin
  unique case (sel)
    2'b00: out = a;
    2'b01: out = b;
    2'b10: out = c;
    2'b11: out = d;
  endcase
end

// priority - first matching case wins (sequential)
always_comb begin
  priority case (1'b1)
    high_priority: out = a;
    med_priority:  out = b;
    low_priority:  out = c;
    default:       out = d;
  endcase
end

// unique0 - at most one match (allows no match)
// priority case (1'b1) - priority encoder pattern
```

### Packed vs Unpacked
```systemverilog
// Packed - contiguous bits, can be sliced
typedef struct packed {
  logic [7:0]  header;
  logic [15:0] payload;
  logic [7:0]  crc;
} packet_t;

// Unpacked - separate storage per element
typedef struct {
  int    count;
  string name;
  real   value;
} config_t;

// Packed array - contiguous bits
logic [3:0][7:0] packed_bytes;  // 32 bits total

// Unpacked array - separate elements
logic [7:0] unpacked_bytes [4];  // 4 separate bytes
```

## Class-Based Patterns

### Class Inheritance
```systemverilog
virtual class base_transaction;
  rand bit [7:0] data;

  pure virtual function void display();

  virtual function bit compare(base_transaction other);
    return (this.data == other.data);
  endfunction
endclass

class extended_transaction extends base_transaction;
  rand bit [7:0] addr;

  function void display();
    $display("addr=%0h data=%0h", addr, data);
  endfunction
endclass
```

### Constraints
```systemverilog
class packet;
  rand bit [7:0]  length;
  rand bit [7:0]  payload[];
  rand bit [15:0] addr;
  rand bit        write;

  // Basic constraint
  constraint c_length { length inside {[1:64]}; }

  // Array size constraint
  constraint c_payload_size { payload.size() == length; }

  // Implication
  constraint c_write_addr { write -> addr[15:12] == 4'hA; }

  // Distribution
  constraint c_dist {
    length dist { [1:8] := 50, [9:64] := 50 };
  }

  // Solve order
  constraint c_order { solve length before payload; }

  // Soft constraint (can be overridden)
  constraint c_soft { soft addr inside {[0:1023]}; }
endclass

// Inline constraint override
packet p = new();
p.randomize() with { length < 10; write == 1; };
```

## Dynamic Data Structures

### Dynamic Arrays
```systemverilog
int dyn_arr[];
dyn_arr = new[10];           // Allocate 10 elements
dyn_arr = new[20](dyn_arr);  // Resize preserving data
dyn_arr.delete();            // Free memory
```

### Queues
```systemverilog
int queue[$];
queue.push_back(1);    // Add to end
queue.push_front(0);   // Add to front
int val = queue.pop_front();  // Remove from front
int size = queue.size();
queue = queue.find(x) with (x > 5);  // Filter
```

### Associative Arrays
```systemverilog
int assoc[string];
assoc["key1"] = 100;
if (assoc.exists("key1")) ...
assoc.delete("key1");
foreach (assoc[key]) $display("%s: %0d", key, assoc[key]);
```

## Synchronization

### Events
```systemverilog
event data_ready;
-> data_ready;           // Trigger
@data_ready;             // Wait for trigger
wait(data_ready.triggered);  // Level-sensitive wait
```

### Semaphores
```systemverilog
semaphore sem = new(1);  // Binary semaphore
sem.get();               // Acquire (blocking)
sem.put();               // Release
if (sem.try_get()) ...   // Non-blocking acquire
```

### Mailboxes
```systemverilog
mailbox #(transaction) mbx = new();
mbx.put(tr);             // Send (blocking if full)
mbx.get(tr);             // Receive (blocking if empty)
mbx.peek(tr);            // Peek without removing
if (mbx.try_get(tr)) ... // Non-blocking receive
```

## Assertion Patterns

### Immediate Assertions
```systemverilog
always_comb begin
  assert (state != INVALID) else $error("Invalid state");

  // With action blocks
  assert (count <= MAX)
    $info("Count OK")
  else
    $fatal("Count overflow");
end
```

### Concurrent Assertions
```systemverilog
// Property with sequence
sequence req_ack_seq;
  req ##[1:5] ack;
endsequence

property req_followed_by_ack;
  @(posedge clk) disable iff (!rst_n)
  req |-> req_ack_seq;
endproperty

assert property (req_followed_by_ack);
cover property (req_followed_by_ack);

// Common operators
// |->  overlapped implication (same cycle)
// |=>  non-overlapped implication (next cycle)
// ##N  fixed delay
// ##[M:N] range delay
// [*N] consecutive repetition
// [=N] non-consecutive repetition
// $rose(), $fell(), $stable(), $past()
```

## Coding Guidelines

1. **Use explicit types**: `logic` over `reg`/`wire` in new code
2. **Use always_ff/always_comb**: Not plain `always`
3. **Avoid latches**: Ensure all paths assign in always_comb
4. **Unique/priority**: Explicitly state case behavior
5. **Packed for bit manipulation**: Use packed structs for protocols
6. **Initialize in declarations**: `logic [7:0] count = 0;`
7. **Use enums for states**: Named states improve readability
8. **Parameterize**: Use parameters for configurable modules
