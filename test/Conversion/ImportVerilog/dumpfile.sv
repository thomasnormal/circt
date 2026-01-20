// RUN: circt-verilog --ir-moore %s | FileCheck %s

module dumpfile_test;
  initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, dumpfile_test);
  end
endmodule

// CHECK: moore.module @dumpfile_test
