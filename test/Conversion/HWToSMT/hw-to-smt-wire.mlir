// RUN: circt-opt %s --convert-hw-to-smt | FileCheck %s

// CHECK-LABEL: func.func @wire_id(%{{.+}}: !smt.bv<8>) -> !smt.bv<8>
hw.module @wire_id(in %in : i8, out out : i8) {
  %w0 = hw.wire %in : i8
  %w1 = hw.wire %w0 name "named_wire" : i8
  hw.output %w1 : i8
}

// CHECK-NOT: hw.wire
// CHECK: return %{{.+}} : !smt.bv<8>
