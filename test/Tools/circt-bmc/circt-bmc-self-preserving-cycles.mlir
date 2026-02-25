// RUN: circt-bmc %s --module=top -b 1 --ignore-asserts-until=0 --emit-mlir | FileCheck %s

// CHECK: func.func @bmc_circuit(%{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<4>, %{{.*}}: !smt.bv<4>) -> !smt.bv<1>
// CHECK-COUNT-2: smt.bv.concat %{{.*}}, %c0_bv1 : !smt.bv<4>, !smt.bv<1>

module {
  hw.module @top(in %in : i1) attributes {num_regs = 0 : i32, initial_values = []} {
    %false = hw.constant false
    %c0_i4 = hw.constant 0 : i4

    // Self-preserving cycle with an OR merge.
    %x_hi = comb.extract %x from 1 : (i5) -> i4
    %x_shift = comb.concat %x_hi, %false : i4, i1
    %x_other = comb.concat %c0_i4, %in : i4, i1
    %x = comb.or %x_shift, %x_other : i5

    // Degenerate self-preserving cycle without an OR merge.
    %y_hi = comb.extract %y from 1 : (i5) -> i4
    %y = comb.concat %y_hi, %false : i4, i1

    // Keep both cycles live through a property.
    %x_msb = comb.extract %x from 4 : (i5) -> i1
    %y_msb = comb.extract %y from 4 : (i5) -> i1
    %p = comb.or %x_msb, %y_msb : i1
    verif.assert %p : i1
    hw.output
  }
}
