// RUN: circt-opt --convert-hw-to-smt %s | FileCheck %s

hw.module.extern @sym(out y : i1)

hw.module @top() {
  %y = hw.instance "u" @sym() -> (y : i1)
  verif.assert %y : i1
  hw.output
}

// CHECK: func.func @top()
// CHECK: smt.declare_fun : !smt.bv<1>
// CHECK-NOT: smt.apply_func
