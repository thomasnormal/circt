// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Dynamic-array assignment must deep-copy payload and size metadata.
// A shallow {ptr,len} copy aliases storage and breaks downstream users
// (e.g. UVM analysis FIFOs and coverage consumers).

module top;
  bit [7:0] a8[];
  bit [7:0] b8[];
  bit [15:0] a16[];
  bit [15:0] b16[];

  initial begin
    a8 = new[1];
    a8[0] = 8'hAB;
    b8 = a8;

    // Mutate and resize the source after assignment.
    a8[0] = 8'h11;
    a8 = new[0];

    // CHECK: b8_size=1 b8_0=ab
    $display("b8_size=%0d b8_0=%0h", b8.size(), b8[0]);

    a16 = new[2];
    a16[0] = 16'hBEEF;
    a16[1] = 16'h1234;
    b16 = a16;

    // Mutate and resize the source after assignment.
    a16[0] = 16'h0000;
    a16[1] = 16'h0000;
    a16 = new[0];

    // CHECK: b16_size=2 b16_0=beef b16_1=1234
    $display("b16_size=%0d b16_0=%0h b16_1=%0h",
             b16.size(), b16[0], b16[1]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
