// RUN: circt-opt --sim-lower-dpi-func %s | FileCheck %s

sim.func.dpi @float_dpi(out arg0: f64, in %arg1: f64)
// CHECK-LABEL: func.func private @float_dpi(!llvm.ptr, f64)
// CHECK-LABEL: func.func @float_dpi_wrapper(%arg0: f64) -> f64 {
// CHECK-NEXT: %0 = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %1 = llvm.alloca %0 x f64 : (i64) -> !llvm.ptr
// CHECK-NEXT: call @float_dpi(%1, %arg0) : (!llvm.ptr, f64) -> ()
// CHECK-NEXT: %2 = llvm.load %1 : !llvm.ptr -> f64
// CHECK-NEXT: return %2 : f64
// CHECK-NEXT: }

hw.module @top(in %clock: !seq.clock, in %in: f64, out out0: f64) {
  // CHECK-LABEL: hw.module @top
  // CHECK: %[[R:.+]] = sim.func.dpi.call @float_dpi_wrapper(%in) clock %clock : (f64) -> f64
  // CHECK: hw.output %[[R]] : f64
  %0 = sim.func.dpi.call @float_dpi(%in) clock %clock : (f64) -> f64
  hw.output %0 : f64
}
