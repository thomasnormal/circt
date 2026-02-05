// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that task-local $readmemh and dynamic mem indexing capture outer refs
// into the generated function so IsolatedFromAbove is satisfied.

module MemLoadTask;
  logic [7:0] mem [0:3];

  task automatic load(input int idx, input string file);
    $readmemh(file, mem);
    mem[idx] = mem[idx + 1];
  endtask

  initial begin
    load(0, "mem.hex");
  end
endmodule

// CHECK-LABEL: func.func private @load
// CHECK: moore.builtin.readmemh %arg{{[0-9]+}}, %arg{{[0-9]+}}
// CHECK: moore.dyn_extract_ref %arg{{[0-9]+}} from %{{[0-9]+}}
