// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-hw %s 2>&1 | FileCheck %s --check-prefix=HW

// Test that X/Z constants are preserved in Moore IR and properly lowered to HW

// MOORE-LABEL: moore.module @XZConstants
// HW-LABEL: hw.module @XZConstants
// HW-COUNT-1: hw.constant -8 : i4
module XZConstants(
  input logic [3:0] in,
  output logic [3:0] out_and,
  output logic [3:0] out_or,
  output logic [3:0] out_xor
);

  // Test AND with X/Z constant
  // b1X00 has X in bit 2, which becomes 0 during lowering
  // MOORE-DAG: moore.constant b1X00 : l4
  // HW: comb.extract {{%.+}} from 3
  assign out_and = in & 4'b1x00;

  // Test OR with X/Z constant
  // b0001 is all known, so OR works normally
  // MOORE-DAG: moore.constant 1 : l4
  // HW: comb.or %{{.+}}, %c1_i4 : i4
  assign out_or = in | 4'b0001;

  // Test XOR with Z constant
  // bZ000 has Z in bit 3, which becomes 0 during lowering
  // MOORE-DAG: moore.constant bZ000 : l4
  // After lowering, XOR with 0 is optimized away
  assign out_xor = in ^ 4'bz000;

endmodule

// MOORE-LABEL: moore.module @AllXZ
// HW-LABEL: hw.module @AllXZ
module AllXZ(
  output logic [3:0] all_x,
  output logic [3:0] all_z
);
  // All X constant - becomes hex X in Moore, uses hw.aggregate_constant in HW
  // MOORE-DAG: moore.constant hX : l4
  // HW-DAG: hw.aggregate_constant
  assign all_x = 4'bxxxx;

  // All Z constant - becomes hex Z in Moore, 0 in HW (reuses same constant)
  // MOORE-DAG: moore.constant hZ : l4
  assign all_z = 4'bzzzz;

endmodule
