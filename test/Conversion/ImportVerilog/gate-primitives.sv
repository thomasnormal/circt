// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test basic gate primitive support (and, or, nand, nor, xor, xnor, buf, not)

// CHECK-LABEL: moore.module @test_and_gate
module test_and_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.and
  and (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_or_gate
module test_or_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.or
  or (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_nand_gate
module test_nand_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.and
  // CHECK: moore.not
  nand (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_nor_gate
module test_nor_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.or
  // CHECK: moore.not
  nor (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_xor_gate
module test_xor_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.xor
  xor (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_xnor_gate
module test_xnor_gate;
  wire a, b, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.xor
  // CHECK: moore.not
  xnor (y, a, b);
endmodule

// CHECK-LABEL: moore.module @test_buf_gate
module test_buf_gate;
  wire a, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  buf (y, a);
endmodule

// CHECK-LABEL: moore.module @test_not_gate
module test_not_gate;
  wire a, y;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.not
  not (y, a);
endmodule

// CHECK-LABEL: moore.module @test_multi_input_and
module test_multi_input_and;
  wire a, b, c, y;
  // Test 3-input AND gate
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: %c = moore.net wire : <l1>
  // CHECK: %y = moore.assigned_variable
  // CHECK: moore.read %a
  // CHECK: moore.read %b
  // CHECK: moore.read %c
  // CHECK: moore.and
  // CHECK: moore.and
  and (y, a, b, c);
endmodule

// CHECK-LABEL: moore.module @test_multi_output_buf
module test_multi_output_buf;
  wire a, y1, y2;
  // Test buffer with multiple outputs
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: %y1 = moore.assigned_variable
  // CHECK: %y2 = moore.assigned_variable
  // CHECK: moore.read %a
  buf (y1, y2, a);
endmodule

// CHECK-LABEL: moore.module @test_named_gates
module test_named_gates;
  wire a, b, y1, y2;
  // Test gates with instance names
  // CHECK: moore.and
  and g1 (y1, a, b);
  // CHECK: moore.or
  or g2 (y2, a, b);
endmodule

// CHECK-LABEL: moore.module @test_bufif1_gate
module test_bufif1_gate;
  wire data_in, enable, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  bufif1 (data_out, data_in, enable);
endmodule

// CHECK-LABEL: moore.module @test_notif1_gate
module test_notif1_gate;
  wire data_in, enable, data_out;
  // CHECK: %data_in = moore.net wire : <l1>
  // CHECK: %data_out = moore.assigned_variable
  // CHECK: moore.read %data_in
  // CHECK: moore.not
  notif1 (data_out, data_in, enable);
endmodule
