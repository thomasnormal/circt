// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s
// Test that $save emits a warning during compilation or simulation.
// Bug: $save is silently dropped — no compile-time or runtime warning.
// IEEE 1800-2017 Section 21.5: $save/$restart are checkpoint/restore tasks.
// An unimplemented $save should at minimum warn the user so they know
// simulation state is NOT being saved.
module top;
  initial begin
    $save("save_checkpoint.dat");
    // CHECK: {{[Ww]arning|[Uu]nsupported|[Uu]nimplemented|[Rr]emark.*save|[Dd]ropp}}
    $display("after_save");
    $finish;
  end
endmodule
