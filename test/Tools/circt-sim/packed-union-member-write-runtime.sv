// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #51: writes through packed union members must update
// the underlying union signal.
module tb;
  union packed { logic [3:0] nibble; logic [3:0] bits; } u;
  initial begin
    u.nibble = 4'hA;
    // CHECK: nibble=a bits=a
    $display("nibble=%h bits=%h", u.nibble, u.bits);
    if (u.nibble === 4'hA) begin
      // CHECK: PASS
      $display("PASS");
    end else begin
      $display("FAIL nibble=%h", u.nibble);
    end
    // CHECK-NOT: FAIL nibble=
    $finish;
  end
endmodule
