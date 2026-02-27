// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// obj.get_randstate() should be a read-only query and must not alter
// subsequent randomization under the same process seed.

class item;
  rand int x;
endclass

module top;
  int v_no_get;
  int v_with_get;
  string state;

  initial begin
    process p = process::self();

    p.srandom(12345);
    begin
      item o1 = new;
      void'(o1.randomize());
      v_no_get = o1.x;
    end

    p.srandom(12345);
    begin
      item o2 = new;
      state = o2.get_randstate();
      void'(o2.randomize());
      v_with_get = o2.x;
    end

    // CHECK: get_randstate_side_effect_free=1
    $display("get_randstate_side_effect_free=%0d",
             (v_no_get == v_with_get));
    $finish;
  end
endmodule
