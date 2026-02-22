// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test inside constraint: value must be within specified set of ranges
module top;
  class item;
    rand int x;
    constraint c { x inside {[1:10], [90:100]}; }

    function new();
      x = 0;
    endfunction
  endclass

  initial begin
    item obj = new();
    int all_in_range;
    int hit_low;
    int hit_high;
    int i;

    all_in_range = 1;
    hit_low = 0;
    hit_high = 0;
    process::self().srandom(42);
    for (i = 0; i < 50; i++) begin
      void'(obj.randomize());
      if (!((obj.x >= 1 && obj.x <= 10) || (obj.x >= 90 && obj.x <= 100)))
        all_in_range = 0;
      if (obj.x >= 1 && obj.x <= 10)
        hit_low = 1;
      if (obj.x >= 90 && obj.x <= 100)
        hit_high = 1;
    end

    // CHECK: all_in_range=1
    $display("all_in_range=%0d", all_in_range);
    // CHECK: hit_both_ranges=1
    $display("hit_both_ranges=%0d", hit_low && hit_high);

    $finish;
  end
endmodule
