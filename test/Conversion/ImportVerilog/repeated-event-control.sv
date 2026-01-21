// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test for repeat loop with event control: repeat(N) @(posedge clk);
// Also tests intra-assignment timing control: a = repeat(N) @(posedge clk) b;

module test_repeated_event;
  logic clk;
  logic [7:0] data, src;
  int count;

  // CHECK-LABEL: moore.procedure initial
  initial begin
    // CHECK: cf.br ^[[CHECK:[a-z0-9]+]](%{{.*}} : !moore.i32)
    // CHECK: ^[[CHECK]](%{{.*}}: !moore.i32):
    // CHECK: [[COND:%[0-9]+]] = moore.bool_cast
    // CHECK: [[COND2:%[0-9]+]] = moore.to_builtin_bool [[COND]]
    // CHECK: cf.cond_br [[COND2]], ^[[BODY:[a-z0-9]+]], ^[[EXIT:[a-z0-9]+]]
    // CHECK: ^[[BODY]]:
    // CHECK: moore.wait_event
    // CHECK: moore.detect_event posedge
    // CHECK: [[NEXT:%[0-9]+]] = moore.sub
    // CHECK: cf.br ^[[CHECK]]([[NEXT]] : !moore.i32)
    // CHECK: ^[[EXIT]]:
    repeat(3) @(posedge clk);
    data = 8'hAB;
  end

  // Test with variable count
  // CHECK-LABEL: moore.procedure always
  always begin
    repeat(count) @(negedge clk);
    data = data + 1;
  end

  // Test intra-assignment repeated event control
  // CHECK-LABEL: moore.procedure always
  always begin
    // The value of src is read BEFORE waiting
    // CHECK: [[SRC_VAL:%[0-9]+]] = moore.read %src
    // CHECK: cf.br ^[[CHECK2:[a-z0-9]+]](%{{.*}} : !moore.i32)
    // CHECK: ^[[CHECK2]](%{{.*}}: !moore.i32):
    // CHECK: moore.bool_cast
    // CHECK: cf.cond_br
    // CHECK: moore.wait_event
    // CHECK: moore.detect_event posedge
    // CHECK: ^[[EXIT2:[a-z0-9]+]]:
    // CHECK: moore.blocking_assign %data, [[SRC_VAL]]
    data = repeat(2) @(posedge clk) src;
  end

endmodule
