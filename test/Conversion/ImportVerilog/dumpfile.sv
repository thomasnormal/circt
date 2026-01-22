// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test VCD dump tasks (IEEE 1800-2017 Section 21.7)
// These should all be silently ignored (with remarks).

module dumpfile_test;
  integer i;
  string fname = "out.vcd";

  initial begin
    // Basic VCD dump tasks
    $dumpfile("waveform.vcd");
    $dumpvars(0, dumpfile_test);
    $dumplimit(1024*1024);

    i = 1;
    #100;
    $dumpoff;
    i = 2;
    #200;
    $dumpon;
    i = 3;
    #100;
    $dumpflush;
    i = 4;
    #100;
    $dumpall;

    // Extended VCD dump port tasks
    $dumpports(dumpfile_test, fname);
    $dumpportslimit(1024*1024, fname);
    $dumpportsoff(fname);
    $dumpportson(fname);
    $dumpportsflush(fname);
    $dumpportsall(fname);
  end
endmodule

// CHECK: moore.module @dumpfile_test
