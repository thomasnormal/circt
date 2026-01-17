// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test distribution constraints in AVIP-style code

// AVIP-style enums
typedef enum bit [1:0] { BYTE = 0, HALFWORD = 1, WORD = 2 } HSize;

class AhbTransaction;
  rand HSize hsizeSeq;
  rand bit [2:0] hburstSeq;
  rand bit [3:0] busyControlSeq[16];

  // Distribution constraint similar to AVIP patterns
  // CHECK: moore.constraint.block @c_hsize
  constraint c_hsize {
    // CHECK: moore.constraint.dist
    hsizeSeq dist { BYTE := 1, HALFWORD := 1, WORD := 1 };
  }

  // Distribution with numeric ranges
  // CHECK: moore.constraint.block @c_hburst
  constraint c_hburst {
    // CHECK: moore.constraint.dist
    hburstSeq dist { 2 := 1, 3 := 1, 4 := 1, 5 := 2, 6 := 2, 7 := 2 };
  }

endclass

module top;
  AhbTransaction txn;
  initial begin
    txn = new;
    if (!txn.randomize()) begin
      $display("Randomization failed");
    end
    $display("hsizeSeq = %0d, hburstSeq = %0d", txn.hsizeSeq, txn.hburstSeq);
  end
endmodule
