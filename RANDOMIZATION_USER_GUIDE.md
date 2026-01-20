# CIRCT Randomization Support - User Guide

## Quick Reference

CIRCT provides comprehensive support for SystemVerilog randomization features commonly used in UVM testbenches.

## Supported Features

### ✅ Random Variables

```systemverilog
class my_transaction;
  rand bit [7:0] data;        // ✅ Random variable
  rand int addr;              // ✅ Random integer
  randc bit [3:0] id;         // ✅ Random cyclic (tracking only)
  bit [7:0] non_rand;         // ✅ Non-random variable
endclass
```

**Status:** Fully supported. `rand` and `randc` modifiers are tracked in the IR.

**Note:** `randc` cyclic behavior (ensuring all values visited before repeat) is not yet enforced in runtime.

### ✅ Class Randomization

```systemverilog
class packet;
  rand bit [31:0] addr;
  rand bit [7:0] data;

  function new();
    addr = 0;
    data = 0;
  endfunction
endclass

module test;
  initial begin
    packet pkt = new();
    int success;

    // Randomize the object
    success = pkt.randomize();

    if (success)
      $display("Randomization succeeded: addr=%h data=%h", pkt.addr, pkt.data);
    else
      $display("Randomization failed");
  end
endmodule
```

**Generated IR:**
```mlir
%pkt = moore.class.new : <@packet>
%success = moore.randomize %pkt : <@packet>  // Returns i1 (success/failure)
```

**Status:** Fully supported. Returns 1 on success, 0 on failure.

### ✅ Scope Randomization (std::randomize)

```systemverilog
module test;
  initial begin
    int x, y;
    bit [7:0] data;

    // Randomize standalone variables
    if (std::randomize(x))
      $display("x randomized: %0d", x);

    // Randomize multiple variables
    if (std::randomize(x, y, data))
      $display("x=%0d y=%0d data=%0d", x, y, data);
  end
endmodule
```

**Generated IR:**
```mlir
%success1 = moore.std_randomize %x : !moore.ref<i32>
%success2 = moore.std_randomize %x, %y, %data : !moore.ref<i32>, !moore.ref<i32>, !moore.ref<l8>
```

**Status:** Fully supported for integer types.

### ✅ Range Constraints

```systemverilog
class constrained_packet;
  rand bit [7:0] addr;
  rand int count;

  // Simple range constraint
  constraint c_addr {
    addr inside {[10:20]};
  }

  // Multi-range constraint
  constraint c_count {
    count inside {[1:10], [100:200]};
  }
endclass
```

**Generated IR:**
```mlir
moore.constraint.block @c_addr {
  %cond = moore.and %uge, %ule : i1  // addr >= 10 && addr <= 20
  moore.constraint.expr %cond : i1
}

moore.constraint.block @c_count {
  // Multi-range represented as array of [min, max] pairs
  // Lowered to runtime call: __moore_randomize_with_ranges([1,10,100,200], 2)
}
```

**Status:** Fully supported. Constraints are extracted during lowering and enforced at runtime.

### ✅ Expression Constraints

```systemverilog
class my_class;
  rand int a, b;

  constraint c1 {
    a > 0;
    a < 100;
    b == a * 2;
  }
endclass
```

**Generated IR:**
```mlir
moore.constraint.block @c1 {
  %gt = moore.sgt %a, %zero : i32 -> i1
  moore.constraint.expr %gt : i1

  %lt = moore.slt %a, %hundred : i32 -> i1
  moore.constraint.expr %lt : i1

  %mul = moore.mul %a, %two : i32
  %eq = moore.eq %b, %mul : i32 -> i1
  moore.constraint.expr %eq : i1
}
```

**Status:** Supported for simple expressions. Complex interdependent constraints may not be optimally solved.

### ✅ Soft Constraints

```systemverilog
class my_class;
  rand bit [7:0] value;

  // Hard constraint - must be satisfied
  constraint c_hard {
    value inside {[0:100]};
  }

  // Soft constraint - preferred but can be overridden
  constraint c_soft {
    soft value == 50;
  }
endclass
```

**Generated IR:**
```mlir
moore.constraint.block @c_hard {
  moore.constraint.expr %cond : i1
}

moore.constraint.block @c_soft {
  moore.constraint.expr %eq : i1 soft  // Marked as soft
}
```

**Status:** Fully supported. Soft constraints only apply when no conflicting hard constraints exist.

### ✅ Implication Constraints

```systemverilog
class my_class;
  rand bit [1:0] mode;
  rand bit [7:0] value;

  constraint c_impl {
    (mode == 2'b00) -> value inside {[0:10]};
    (mode == 2'b01) -> value inside {[20:30]};
  }
endclass
```

**Generated IR:**
```mlir
moore.constraint.block @c_impl {
  moore.constraint.implication %cond1 {
    moore.constraint.expr %range1 : i1
  }
  moore.constraint.implication %cond2 {
    moore.constraint.expr %range2 : i1
  }
}
```

**Status:** Supported. Antecedent evaluated first, consequent only checked if antecedent true.

### ✅ If-Else Constraints

```systemverilog
class my_class;
  rand bit mode;
  rand bit [7:0] value;

  constraint c_if {
    if (mode) {
      value inside {[0:50]};
    } else {
      value inside {[51:100]};
    }
  }
endclass
```

**Status:** Supported via `moore.constraint.ifelse` operation.

## UVM-Specific Examples

### Sequence Item Randomization

```systemverilog
class my_seq_item extends uvm_sequence_item;
  rand bit [31:0] addr;
  rand bit [7:0] data;
  rand bit write;

  constraint c_addr {
    addr inside {[32'h1000:32'h1FFF]};
  }

  constraint c_write_bias {
    soft write == 1;  // Prefer writes, but allow reads
  }
endclass

// Usage in sequence:
class my_sequence extends uvm_sequence#(my_seq_item);
  task body();
    my_seq_item item;
    item = my_seq_item::type_id::create("item");

    repeat(10) begin
      if (!item.randomize()) begin
        `uvm_error("SEQ", "Randomization failed")
      end
      start_item(item);
      finish_item(item);
    end
  endtask
endclass
```

**Status:** ✅ Works as expected. All constraints honored.

### Constrained Random Stimulus

```systemverilog
class packet extends uvm_sequence_item;
  rand bit [7:0] length;
  rand bit [7:0] payload[];

  constraint c_length {
    length inside {[1:64]};
  }

  constraint c_payload_size {
    payload.size == length;
  }

  constraint c_payload_values {
    foreach (payload[i]) {
      payload[i] inside {[0:255]};
    }
  }
endclass
```

**Status:**
- ✅ Range constraints on `length`
- ✅ Array size constraints
- ✅ Foreach constraints on array elements

## Runtime Implementation

### Randomization Strategy

CIRCT uses a two-phase randomization approach:

**Phase 1: Basic Randomization**
```c
int32_t __moore_randomize_basic(void *classPtr, int64_t classSize) {
  // Fill entire class memory with random bits
  // Fast, uses __moore_urandom() for efficiency
}
```

**Phase 2: Constraint Application**
```c
// Single range constraint
int64_t __moore_randomize_with_range(int64_t min, int64_t max);

// Multi-range constraint
int64_t __moore_randomize_with_ranges(int64_t *ranges, int64_t numRanges);

// Modulo constraint
int64_t __moore_randomize_with_modulo(int64_t mod, int64_t remainder);
```

### Performance Characteristics

- **Basic randomization:** O(classSize) - very fast
- **Range constraints:** O(1) per constraint - fast random selection from range
- **Multi-range constraints:** O(numRanges) - efficient
- **Expression constraints:** O(numConstraints) - evaluated and enforced

**Note:** This is not a full constraint solver (like SAT/SMT-based solvers in commercial simulators). Complex interdependent constraints may not be optimally solved, but most UVM patterns work well.

## Limitations and Workarounds

### ❌ Builtin Methods Not Supported

```systemverilog
class my_class;
  rand bit [7:0] value;

  // ❌ Not supported yet
  function void pre_randomize();
    $display("About to randomize");
  endfunction

  // ❌ Not supported yet
  function void post_randomize();
    $display("Randomized: value=%0d", value);
  endfunction
endclass
```

**Workaround:** Call pre/post actions manually:
```systemverilog
my_class obj = new();
pre_action();              // Manual pre-randomize
void'(obj.randomize());
post_action(obj);          // Manual post-randomize
```

### ⚠️ randc Cyclic Behavior Not Enforced

```systemverilog
class my_class;
  randc bit [3:0] id;  // ✅ Tracked, but cyclic behavior not enforced
endclass
```

**Status:** The `randc` modifier is recognized and tracked in the IR, but the runtime doesn't yet ensure all values are visited before repeating.

**Workaround:** Implement cyclic behavior manually if critical:
```systemverilog
class my_class;
  rand bit [3:0] id;
  bit [3:0] used_ids[$];

  constraint c_unique {
    !(id inside {used_ids});
  }

  function void post_randomize();
    used_ids.push_back(id);
    if (used_ids.size() == 16)
      used_ids.delete();  // Reset when all values used
  endfunction
endclass
```

### ❌ Dynamic Constraint Control Not Supported

```systemverilog
class my_class;
  rand bit [7:0] value;
  constraint c1 { value < 50; }

  function void setup();
    c1.constraint_mode(0);  // ❌ Not supported
    value.rand_mode(0);     // ❌ Not supported
  endfunction
endclass
```

**Workaround:** Use separate classes or conditional constraints:
```systemverilog
class my_class;
  rand bit [7:0] value;
  rand bit use_constraint;

  constraint c1 {
    use_constraint -> value < 50;
  }
endclass
```

## Debugging Tips

### Check Randomization Success

Always check the return value:
```systemverilog
if (!obj.randomize()) begin
  `uvm_error("RAND", "Randomization failed")
  return;
end
```

### Inspect Generated IR

Use `--ir-moore` to see how constraints are represented:
```bash
circt-verilog --ir-moore your_file.sv | grep -A10 "constraint.block"
```

### Verify Constraint Extraction

Check the lowered code to see if constraints are being extracted:
```bash
circt-verilog --ir-moore your_file.sv | grep -E "randomize|constraint"
```

## Best Practices

### 1. Keep Constraints Simple

✅ **Good:**
```systemverilog
constraint c1 {
  value inside {[0:100]};
  count inside {[1:10]};
}
```

❌ **Avoid (may not be well-supported):**
```systemverilog
constraint c1 {
  // Complex interdependent constraints
  value1 * value2 + value3 == value4;
  value5 % 7 == value6 % 11;
}
```

### 2. Use Soft Constraints for Preferences

```systemverilog
// Hard constraint - must be satisfied
constraint c_valid {
  addr inside {[32'h1000:32'hFFFF]};
}

// Soft constraint - preferred value
constraint c_default {
  soft addr == 32'h1000;
}
```

### 3. Leverage Implication Constraints

```systemverilog
constraint c_mode {
  (write_mode) -> addr inside {[32'h1000:32'h1FFF]};
  (!write_mode) -> addr inside {[32'h2000:32'h2FFF]};
}
```

### 4. Test Randomization Coverage

```systemverilog
// Ensure randomization reaches all expected ranges
for (int i = 0; i < 1000; i++) begin
  assert(obj.randomize());
  // Collect coverage
end
```

## Migration from Commercial Simulators

### What Works the Same

✅ Basic rand/randc declarations
✅ Range constraints with `inside`
✅ Expression constraints
✅ Soft constraints
✅ Implication constraints
✅ Success/failure return values

### What's Different

⚠️ No pre_randomize()/post_randomize() hooks - call manually
⚠️ No constraint_mode()/rand_mode() - use conditional constraints
⚠️ randc doesn't enforce cyclic behavior - implement manually if needed
⚠️ Constraint solver may not handle very complex constraints optimally

### Example Migration

**Before (commercial simulator):**
```systemverilog
class my_class;
  rand bit [7:0] value;
  constraint c1 { value < 50; }

  function void pre_randomize();
    $display("Pre");
  endfunction

  function void post_randomize();
    $display("Post: %0d", value);
  endfunction
endclass

my_class obj = new();
obj.c1.constraint_mode(0);  // Disable constraint
obj.randomize();
```

**After (CIRCT):**
```systemverilog
class my_class;
  rand bit [7:0] value;
  rand bit use_c1;
  constraint c1 { use_c1 -> value < 50; }
endclass

my_class obj = new();
obj.use_c1 = 0;  // Control constraint via variable
$display("Pre");
obj.randomize();
$display("Post: %0d", obj.value);
```

## Summary

CIRCT provides **production-ready randomization support** for most UVM testbenches:

✅ **Strengths:**
- Fast randomization runtime
- Comprehensive constraint support (range, expression, soft, implication)
- IEEE 1800-2017 compliant API
- Good performance for typical UVM patterns

⚠️ **Limitations:**
- No pre_randomize()/post_randomize() hooks
- No dynamic constraint_mode()/rand_mode()
- randc cyclic behavior not enforced
- Complex interdependent constraints may not be optimally solved

For typical UVM verification workflows, these limitations are minor and can be worked around if needed.

---

**For More Information:**
- See `ITERATION_45_TRACK_B_SUMMARY.md` for detailed implementation notes
- Check `test/Conversion/ImportVerilog/class-randomization*.sv` for examples
- Review Moore dialect operations: `include/circt/Dialect/Moore/MooreOps.td`
