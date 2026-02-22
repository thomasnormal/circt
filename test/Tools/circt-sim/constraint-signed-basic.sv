// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test mixed signedness in constraint expressions:
// - signed int with negative bounds
module top;
  class item;
    rand int x;
    constraint c { x >= -50; x <= -10; }

    function new();
      x = 0;
    endfunction
  endclass

  initial begin
    item obj = new();
    int all_in_range;
    int i;

    all_in_range = 1;
    process::self().srandom(42);
    for (i = 0; i < 50; i++) begin
      void'(obj.randomize());
      if (obj.x < -50 || obj.x > -10)
        all_in_range = 0;
    end

    // CHECK: all_in_range=1
    $display("all_in_range=%0d", all_in_range);

    $finish;
  end
endmodule
