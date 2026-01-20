// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Constraint Implication Operator Tests - Iteration 62 Track B
// Tests for -> implication and if-else conditional constraint operators.
// IEEE 1800-2017 Section 18.5.6 "Implication constraints"
// IEEE 1800-2017 Section 18.5.7 "if-else constraints"
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Basic -> implication operator
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @basic_implication {
// CHECK:   moore.class.propertydecl @mode : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_basic_impl {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class basic_implication;
  rand bit [1:0] mode;  // 0=IDLE, 1=READ, 2=WRITE
  rand bit [7:0] data;

  // When mode is READ (1), data must be in range [0:127]
  constraint c_basic_impl {
    (mode == 1) -> (data < 128);
  }

  function new();
    mode = 0;
    data = 0;
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 2: Multiple -> implications in one constraint block
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @multi_implication {
// CHECK:   moore.class.propertydecl @op_type : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @addr : !moore.i16 rand_mode rand
// CHECK:   moore.constraint.block @c_multi_impl {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class multi_implication;
  rand bit [1:0] op_type;  // 0=NOP, 1=READ, 2=WRITE
  rand bit [15:0] addr;

  constraint c_multi_impl {
    // Read operations must have addr < 0x1000
    (op_type == 1) -> (addr < 16'h1000);
    // Write operations must have addr >= 0x1000
    (op_type == 2) -> (addr >= 16'h1000);
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 3: Nested -> implications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @nested_implication {
// CHECK:   moore.class.propertydecl @enable : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @mode : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @value : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_nested {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// Note: Nested implication is optimized by slang: (a -> (b -> c)) = (a -> (!b | c))
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class nested_implication;
  rand bit enable;
  rand bit mode;
  rand bit [7:0] value;

  // Nested implication: enable -> (mode -> value > 100)
  // Only when both enable and mode are true, value must be > 100
  // Note: slang optimizes (a -> (b -> c)) to (a -> (!b | c))
  constraint c_nested {
    enable -> (mode -> (value > 100));
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 4: Basic if-else constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @basic_ifelse {
// CHECK:   moore.class.propertydecl @is_write : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @size : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_basic_ifelse {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     } else {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class basic_ifelse;
  rand bit is_write;
  rand bit [7:0] size;

  constraint c_basic_ifelse {
    if (is_write) {
      size inside {[1:64]};
    } else {
      size inside {[1:256]};
    }
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 5: if-else without else branch
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @if_only {
// CHECK:   moore.class.propertydecl @valid : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_if_only {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class if_only;
  rand bit valid;
  rand int data;

  // When valid, data must be non-negative
  constraint c_if_only {
    if (valid) {
      data >= 0;
    }
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 6: Nested if-else constraints
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @nested_ifelse {
// CHECK:   moore.class.propertydecl @level : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @prio_val : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_nested_ifelse {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     } else {
// CHECK:       moore.constraint.if_else {{.*}} : i1 {
// CHECK:         moore.constraint.expr {{.*}} : i1
// CHECK:       } else {
// CHECK:         moore.constraint.expr {{.*}} : i1
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

class nested_ifelse;
  rand bit [1:0] level;  // 0=LOW, 1=MEDIUM, 2=HIGH
  rand bit [7:0] prio_val;

  constraint c_nested_ifelse {
    if (level == 0) {
      prio_val inside {[0:50]};
    } else if (level == 1) {
      prio_val inside {[51:150]};
    } else {
      prio_val inside {[151:255]};
    }
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 7: Soft constraint with implication
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @soft_implication {
// CHECK:   moore.class.propertydecl @enable : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @threshold : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_soft_impl {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1 soft
// CHECK:     }
// CHECK:   }
// CHECK: }

class soft_implication;
  rand bit enable;
  rand int threshold;

  // Soft constraint within implication - default threshold when enabled
  constraint c_soft_impl {
    enable -> soft threshold == 100;
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 8: Soft constraint with if-else
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @soft_ifelse {
// CHECK:   moore.class.propertydecl @mode : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_soft_ifelse {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1 soft
// CHECK:     } else {
// CHECK:       moore.constraint.expr {{.*}} : i1 soft
// CHECK:     }
// CHECK:   }
// CHECK: }

class soft_ifelse;
  rand bit mode;
  rand int value;

  // Soft defaults for both branches
  constraint c_soft_ifelse {
    if (mode) {
      soft value == 1000;
    } else {
      soft value == 0;
    }
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 9: Mixed -> and if-else in same block
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @mixed_impl_ifelse {
// CHECK:   moore.class.propertydecl @type_sel : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @len : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @start_addr : !moore.i16 rand_mode rand
// CHECK:   moore.constraint.block @c_mixed {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     } else {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class mixed_impl_ifelse;
  rand bit [1:0] type_sel;
  rand bit [7:0] len;
  rand bit [15:0] start_addr;

  constraint c_mixed {
    // Implication: type_sel == 0 implies short length
    (type_sel == 0) -> (len < 16);

    // if-else for start_addr
    if (type_sel == 1) {
      start_addr inside {[16'h0000:16'h0FFF]};
    } else {
      start_addr inside {[16'h1000:16'hFFFF]};
    }
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 10: Implication with inside expression
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @impl_with_inside {
// CHECK:   moore.class.propertydecl @cmd : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @payload : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_impl_inside {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class impl_with_inside;
  rand bit [1:0] cmd;
  rand bit [7:0] payload;

  constraint c_impl_inside {
    // Complex antecedent with inside
    (cmd inside {1, 2}) -> (payload inside {[10:100]});
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 11: Chained implications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @chained_impl {
// CHECK:   moore.class.propertydecl @a : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @b : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @c : !moore.i1 rand_mode rand
// CHECK:   moore.constraint.block @c_chain {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class chained_impl;
  rand bit a, b, c;

  // Chain: a -> b, b -> c (effectively: a -> b -> c)
  constraint c_chain {
    a -> b;
    b -> c;
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 12: Bidirectional implication (iff pattern)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @bidir_impl {
// CHECK:   moore.class.propertydecl @x : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @y : !moore.i1 rand_mode rand
// CHECK:   moore.constraint.block @c_bidir {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class bidir_impl;
  rand bit x, y;

  // Bidirectional: x iff y (x -> y) and (y -> x)
  constraint c_bidir {
    x -> y;
    y -> x;
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 13: UVM-style transaction constraints with implication
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @uvm_transaction {
// CHECK:   moore.class.propertydecl @kind : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @addr : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @burst_len : !moore.i4 rand_mode rand
// CHECK:   moore.constraint.block @c_kind {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK:   moore.constraint.block @c_addr {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     } else {
// CHECK:       moore.constraint.if_else {{.*}} : i1 {
// CHECK:         moore.constraint.expr {{.*}} : i1
// CHECK:       } else {
// CHECK:         moore.constraint.expr {{.*}} : i1
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK:   moore.constraint.block @c_burst {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class uvm_transaction;
  rand bit [1:0] kind;  // 0=IDLE, 1=READ, 2=WRITE, 3=BURST
  rand bit [31:0] addr;
  rand bit [31:0] data;
  rand bit [3:0] burst_len;

  // Kind must be valid
  constraint c_kind {
    kind inside {[0:3]};
  }

  // Address ranges depend on transaction kind
  constraint c_addr {
    if (kind == 1) {  // READ
      addr inside {[32'h0000_0000:32'h0000_FFFF]};
    } else if (kind == 2) {  // WRITE
      addr inside {[32'h0001_0000:32'h0001_FFFF]};
    } else {
      addr inside {[32'h0002_0000:32'h0002_FFFF]};
    }
  }

  // Burst length only matters for BURST transactions
  constraint c_burst {
    (kind == 3) -> (burst_len inside {[1:15]});
  }

  function new();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test module instantiating all constraint classes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_implication_constraints
module test_implication_constraints;
  initial begin
    automatic basic_implication t1 = new();
    automatic multi_implication t2 = new();
    automatic nested_implication t3 = new();
    automatic basic_ifelse t4 = new();
    automatic if_only t5 = new();
    automatic nested_ifelse t6 = new();
    automatic soft_implication t7 = new();
    automatic soft_ifelse t8 = new();
    automatic mixed_impl_ifelse t9 = new();
    automatic impl_with_inside t10 = new();
    automatic chained_impl t11 = new();
    automatic bidir_impl t12 = new();
    automatic uvm_transaction t13 = new();

    // CHECK: moore.randomize
    void'(t1.randomize());
    $display("t1: mode=%0d, data=%0d", t1.mode, t1.data);

    // CHECK: moore.randomize
    void'(t2.randomize());
    $display("t2: op_type=%0d, addr=%0h", t2.op_type, t2.addr);

    // CHECK: moore.randomize
    void'(t3.randomize());
    $display("t3: enable=%0d, mode=%0d, value=%0d", t3.enable, t3.mode, t3.value);

    // CHECK: moore.randomize
    void'(t4.randomize());
    $display("t4: is_write=%0d, size=%0d", t4.is_write, t4.size);

    // CHECK: moore.randomize
    void'(t5.randomize());
    $display("t5: valid=%0d, data=%0d", t5.valid, t5.data);

    // CHECK: moore.randomize
    void'(t6.randomize());
    $display("t6: level=%0d, prio_val=%0d", t6.level, t6.prio_val);

    // CHECK: moore.randomize
    void'(t7.randomize());
    $display("t7: enable=%0d, threshold=%0d", t7.enable, t7.threshold);

    // CHECK: moore.randomize
    void'(t8.randomize());
    $display("t8: mode=%0d, value=%0d", t8.mode, t8.value);

    // CHECK: moore.randomize
    void'(t9.randomize());
    $display("t9: type_sel=%0d, len=%0d, start_addr=%0h", t9.type_sel, t9.len, t9.start_addr);

    // CHECK: moore.randomize
    void'(t10.randomize());
    $display("t10: cmd=%0d, payload=%0d", t10.cmd, t10.payload);

    // CHECK: moore.randomize
    void'(t11.randomize());
    $display("t11: a=%0d, b=%0d, c=%0d", t11.a, t11.b, t11.c);

    // CHECK: moore.randomize
    void'(t12.randomize());
    $display("t12: x=%0d, y=%0d (should be equal)", t12.x, t12.y);

    // CHECK: moore.randomize
    void'(t13.randomize());
    $display("t13: kind=%0d, addr=%0h, data=%0h, burst_len=%0d",
             t13.kind, t13.addr, t13.data, t13.burst_len);
  end
endmodule
