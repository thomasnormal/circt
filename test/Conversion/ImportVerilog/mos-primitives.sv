// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test MOS transistor primitive support (nmos, pmos, cmos, tran, tranif)

//===----------------------------------------------------------------------===//
// Basic MOS Switches: nmos, pmos, rnmos, rpmos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_nmos
module test_nmos;
  wire data_in, control, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  nmos (data_out, data_in, control);
endmodule

// CHECK-LABEL: moore.module @test_pmos
module test_pmos;
  wire data_in, control, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  pmos (data_out, data_in, control);
endmodule

// CHECK-LABEL: moore.module @test_rnmos
module test_rnmos;
  wire data_in, control, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  rnmos (data_out, data_in, control);
endmodule

// CHECK-LABEL: moore.module @test_rpmos
module test_rpmos;
  wire data_in, control, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  rpmos (data_out, data_in, control);
endmodule

// CHECK-LABEL: moore.module @test_nmos_named
module test_nmos_named;
  wire d, c, q;
  // CHECK: %d = moore.net wire : <l1>
  // CHECK: %q = moore.assigned_variable
  nmos n1 (q, d, c);
endmodule

//===----------------------------------------------------------------------===//
// Complementary MOS: cmos, rcmos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cmos
module test_cmos;
  wire data_in, nctrl, pctrl, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  cmos (data_out, data_in, nctrl, pctrl);
endmodule

// CHECK-LABEL: moore.module @test_rcmos
module test_rcmos;
  wire data_in, nctrl, pctrl, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  rcmos (data_out, data_in, nctrl, pctrl);
endmodule

// CHECK-LABEL: moore.module @test_cmos_named
module test_cmos_named;
  wire d, nc, pc, q;
  // CHECK: %d = moore.net wire : <l1>
  // CHECK: %q = moore.assigned_variable
  cmos c1 (q, d, nc, pc);
endmodule

//===----------------------------------------------------------------------===//
// Bidirectional Switches: tran, rtran
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_tran
module test_tran;
  wire a, b;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  tran (a, b);
endmodule

// CHECK-LABEL: moore.module @test_rtran
module test_rtran;
  wire a, b;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  rtran (a, b);
endmodule

// CHECK-LABEL: moore.module @test_tran_named
module test_tran_named;
  wire x, y;
  // CHECK: %x = moore.assigned_variable %y
  // CHECK: %y = moore.assigned_variable %x
  tran t1 (x, y);
endmodule

//===----------------------------------------------------------------------===//
// Controlled Bidirectional Switches: tranif0, tranif1, rtranif0, rtranif1
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_tranif0
module test_tranif0;
  wire a, b, ctrl;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  tranif0 (a, b, ctrl);
endmodule

// CHECK-LABEL: moore.module @test_tranif1
module test_tranif1;
  wire a, b, ctrl;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  tranif1 (a, b, ctrl);
endmodule

// CHECK-LABEL: moore.module @test_rtranif0
module test_rtranif0;
  wire a, b, ctrl;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  rtranif0 (a, b, ctrl);
endmodule

// CHECK-LABEL: moore.module @test_rtranif1
module test_rtranif1;
  wire a, b, ctrl;
  // CHECK: %a = moore.assigned_variable %b
  // CHECK: %b = moore.assigned_variable %a
  rtranif1 (a, b, ctrl);
endmodule

// CHECK-LABEL: moore.module @test_tranif1_named
module test_tranif1_named;
  wire p, q, en;
  // CHECK: %p = moore.assigned_variable %q
  // CHECK: %q = moore.assigned_variable %p
  tranif1 t1 (p, q, en);
endmodule

//===----------------------------------------------------------------------===//
// Multibit MOS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_nmos_multibit
module test_nmos_multibit;
  wire [7:0] data_in, data_out;
  wire control;
  // CHECK: %data_in = moore.net wire : <l8>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  nmos (data_out, data_in, control);
endmodule

// CHECK-LABEL: moore.module @test_cmos_multibit
module test_cmos_multibit;
  wire [15:0] data_in, data_out;
  wire nctrl, pctrl;
  // CHECK: %data_in = moore.net wire : <l16>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  cmos (data_out, data_in, nctrl, pctrl);
endmodule
