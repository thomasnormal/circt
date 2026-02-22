// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd | circt-sim --top top --mode=interpret | FileCheck %s
// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd | circt-sim --top top --mode=compile --jit-compile-budget=0 | FileCheck %s

module top;
  byte src[];
  byte dst[];

  task automatic copy_after_wait();
    #1;
    dst = new[src.size()](src);
  endtask

  initial begin
    src = new[2];
    src[0] = 8'h11;
    src[1] = 8'h22;
    fork
      copy_after_wait();
    join
    $display("dst0=%0h dst1=%0h sz=%0d", dst[0], dst[1], dst.size());
    $finish;
  end
endmodule

// CHECK: dst0=11 dst1=22 sz=2
