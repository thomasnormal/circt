// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK: func.func @lower_assert([[ARG0:%.+]]: i1)
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK-DAG:   [[Cn1_BV:%.+]] = smt.bv.constant #smt.bv<-1>
// CHECK:   [[EQ:%.+]] = smt.eq [[CAST]], [[Cn1_BV]]
// CHECK:   [[NEQ:%.+]] = smt.not [[EQ]]
// CHECK:   smt.assert [[NEQ]]
// CHECK:   return

func.func @lower_assert(%arg0: i1) {
  verif.assert %arg0 : i1
  return
}

// CHECK: func.func @lower_assume([[ARG0:%.+]]: i1)
// CHECK-DAG:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK-DAG:   [[Cn1_BV:%.+]] = smt.bv.constant #smt.bv<-1>
// CHECK:   [[EQ:%.+]] = smt.eq [[CAST]], [[Cn1_BV]]
// CHECK:   smt.assert [[EQ]]
// CHECK:   return

func.func @lower_assume(%arg0: i1) {
  verif.assume %arg0 : i1
  return
}

// CHECK-LABEL: func @test_lec
// CHECK-SAME:  ([[ARG0:%.+]]: !smt.bv<1>)
func.func @test_lec(%arg0: !smt.bv<1>) -> (i1, i1, i1) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
  // CHECK: [[C0:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
  // CHECK: [[V0:%.+]] = smt.eq %arg0, [[C0]] : !smt.bv<1>
  // CHECK: [[V1:%.+]] = smt.not [[V0]]
  // CHECK: smt.assert [[V1]]
  verif.assert %0 : i1

  // CHECK: smt.solver() : () -> i1
  // CHECK-DAG: [[IN0:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-DAG: [[IN1:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-DAG: [[V0:%.+]] = builtin.unrealized_conversion_cast [[IN0]] : !smt.bv<32> to i32
  // CHECK-DAG: [[V1:%.+]] = builtin.unrealized_conversion_cast [[IN1]] : !smt.bv<32> to i32
  // CHECK: [[V2:%.+]]:2 = "some_op"([[V0]], [[V1]]) : (i32, i32) -> (i32, i32)
  // CHECK-DAG: [[V3:%.+]] = builtin.unrealized_conversion_cast [[V2]]#0 : i32 to !smt.bv<32>
  // CHECK-DAG: [[C1OUT0:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<32>
  // CHECK-DAG: [[C2OUT0:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<32>
  // CHECK: smt.eq [[C1OUT0]], [[IN0]]
  // CHECK: smt.eq [[C2OUT0]], [[V3]]
  // CHECK-DAG: [[V5:%.+]] = builtin.unrealized_conversion_cast [[V2]]#1 : i32 to !smt.bv<32>
  // CHECK-DAG: [[C1OUT1:%.+]] = smt.declare_fun "c1_out1" : !smt.bv<32>
  // CHECK-DAG: [[C2OUT1:%.+]] = smt.declare_fun "c2_out1" : !smt.bv<32>
  // CHECK: smt.eq [[C1OUT1]], [[IN1]]
  // CHECK: smt.eq [[C2OUT1]], [[V5]]
  // CHECK-DAG: [[D0:%.+]] = smt.distinct [[C1OUT0]], [[C2OUT0]] : !smt.bv<32>
  // CHECK-DAG: [[D1:%.+]] = smt.distinct [[C1OUT1]], [[C2OUT1]] : !smt.bv<32>
  // CHECK-DAG: [[V7:%.+]] = smt.or [[D0]], [[D1]]
  // CHECK: smt.assert [[V7]]
  // CHECK-DAG: [[FALSE:%.+]] = arith.constant false
  // CHECK-DAG: [[TRUE:%.+]] = arith.constant true
  // CHECK: [[V8:%.+]] = smt.check
  // CHECK: smt.yield [[FALSE]]
  // CHECK: smt.yield [[FALSE]]
  // CHECK: smt.yield [[TRUE]]
  // CHECK: smt.yield [[V8]] :
  %1 = verif.lec : i1 first {
  ^bb0(%arg1: i32, %arg2: i32):
    verif.yield %arg1, %arg2 : i32, i32
  } second {
  ^bb0(%arg1: i32, %arg2: i32):
    %2, %3 = "some_op"(%arg1, %arg2) : (i32, i32) -> (i32, i32)
    verif.yield %2, %3 : i32, i32
  }

  %2 = verif.lec : i1  first {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  } second {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  }

  %3 = verif.lec : i1 first {
  ^bb0(%arg1: i32):
    verif.yield
  } second {
  ^bb0(%arg1: i32):
    verif.yield
  }

  verif.lec first {
  ^bb0(%arg1: i32):
    verif.yield
  } second {
  ^bb0(%arg1: i32):
    verif.yield
  }

  verif.lec first {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  } second {
  ^bb0(%arg2: i32):
    verif.yield %arg2 : i32
  }
  // CHECK: smt.solver() : () -> () {
  // CHECK:   [[V11:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK:   [[C1OUTX:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<32>
  // CHECK:   [[C2OUTX:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<32>
  // CHECK:   [[EQ1X:%.+]] = smt.eq [[C1OUTX]], [[V11]] : !smt.bv<32>
  // CHECK:   smt.assert [[EQ1X]]
  // CHECK:   [[EQ2X:%.+]] = smt.eq [[C2OUTX]], [[V11]] : !smt.bv<32>
  // CHECK:   smt.assert [[EQ2X]]
  // CHECK:   [[EQ3:%.+]] = smt.distinct [[C1OUTX]], [[C2OUTX]] : !smt.bv<32>
  // CHECK:   smt.assert [[EQ3]]
  // CHECK:   smt.check sat {
  // CHECK:   } unknown {
  // CHECK:   } unsat {
  // CHECK:   }
  // CHECK: }

  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}} : i1, i1, i1
  return %1, %2, %3 : i1, i1, i1
}

// Test BMC lowering: circuit is called, then loop, then edge detection for regs
// CHECK-LABEL:  func.func @test_bmc() -> i1 {
// CHECK:    [[BMC:%.+]] = smt.solver
// CHECK-DAG:  [[CNEG1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-DAG:  [[TRUE:%.+]] = arith.constant true
// CHECK-DAG:  [[FALSE:%.+]] = arith.constant false
// CHECK-DAG:  [[C10_I32:%.+]] = arith.constant 10 : i32
// CHECK-DAG:  [[C1_I32:%.+]] = arith.constant 1 : i32
// CHECK-DAG:  [[C0_I32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:  [[C42_BV32:%.+]] = smt.bv.constant #smt.bv<42> : !smt.bv<32>
// CHECK-DAG:  [[INIT:%.+]]:2 = func.call @bmc_init()
// CHECK-DAG:  smt.declare_fun : !smt.bv<32>
// CHECK-DAG:  smt.declare_fun : !smt.bv<32>
// CHECK-DAG:  smt.declare_fun : !smt.array<[!smt.bv<1> -> !smt.bv<32>]>
// CHECK:      scf.for
// The circuit now returns 5 values (4 outputs + 1 property)
// CHECK:        func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.array<[!smt.bv<1> -> !smt.bv<32>]>, !smt.bv<1>)
// Loop is called after circuit
// CHECK:        func.call @bmc_loop
// Edge detection for register updates
// CHECK:        smt.bv.not
// CHECK:        smt.bv.and
// CHECK:        smt.eq {{%.+}}, [[CNEG1]] : !smt.bv<1>
// Property checking
// CHECK:        smt.eq {{%.+}}, [[CNEG1]] : !smt.bv<1>
// CHECK:        smt.not
// CHECK:        smt.and
// CHECK:        smt.push 1
// CHECK:        smt.assert
// CHECK:        smt.check sat
// CHECK:        smt.pop 1
// CHECK:        arith.ori
// Register updates with smt.ite
// CHECK:        smt.ite
// CHECK:        smt.ite
// CHECK:        smt.ite
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      arith.xori
// CHECK:      smt.yield
// CHECK:    }
// CHECK:    return [[BMC]]
// CHECK:  }

// RUN: circt-opt %s --convert-verif-to-smt="rising-clocks-only=true" --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=CHECK1
// CHECK1-LABEL:  func.func @test_bmc() -> i1 {
// CHECK1:        func.call @bmc_circuit
// CHECK1-SAME: -> (!smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.array<[!smt.bv<1> -> !smt.bv<32>]>, !smt.bv<1>)
// In rising-clocks-only mode, bmc_loop is called then the check is done
// CHECK1:        func.call @bmc_loop
// CHECK1:        smt.eq
// CHECK1:        smt.not
// CHECK1:        smt.push 1
// CHECK1:        smt.assert
// CHECK1:        smt.check
// CHECK1:        smt.pop 1
// CHECK1:        arith.ori
// CHECK1:        scf.yield

func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 3 initial_values [unit, 42 : i32, unit]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk, %c0_i1 : !seq.clock, i1
  }
  loop {
    ^bb0(%clk: !seq.clock, %stateArg: i1):
    %from_clock = seq.from_clock %clk
    %c-1_i1 = hw.constant -1 : i1
    %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
    %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
    %newclk = seq.to_clock %neg_clock
    verif.yield %newclk, %newStateArg : !seq.clock, i1
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32, %state1: i32, %state2: !hw.array<2xi32>):
    %true = hw.constant true
    verif.assert %true : i1
    %c-1_i32 = hw.constant -1 : i32
    %0 = comb.add %arg0, %state0 : i32
    // %state0 is the result of a seq.compreg taking %0 as input
    %2 = comb.xor %state0, %c-1_i32 : i32
    verif.yield %2, %0, %state1, %state2 : i32, i32, i32, !hw.array<2xi32>
  }
  func.return %bmc : i1
}

// CHECK-LABEL:  func.func @bmc_init() -> (!smt.bv<1>, !smt.bv<1>) {
// CHECK:    seq.const_clock{{ *}}low
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    return
// CHECK:  }
// CHECK:  func.func @bmc_loop({{%.+}}: !smt.bv<1>, {{%.+}}: !smt.bv<1>)
// CHECK:    hw.constant true
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    seq.from_clock
// CHECK:    comb.xor
// CHECK:    comb.xor
// CHECK:    seq.to_clock
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    builtin.unrealized_conversion_cast
// CHECK:    return
// CHECK:  }
// CHECK:  func.func @bmc_circuit({{%.+}}: !smt.bv<1>, {{%.+}}: !smt.bv<32>, {{%.+}}: !smt.bv<32>, {{%.+}}: !smt.bv<32>, {{%.+}}: !smt.array<[!smt.bv<1> -> !smt.bv<32>]>) -> (!smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.array<[!smt.bv<1> -> !smt.bv<32>]>, !smt.bv<1>)
// CHECK-DAG:    hw.constant true
// CHECK-DAG:    hw.constant -1 : i32
// CHECK:    comb.add
// CHECK:    comb.xor
// CHECK:    return
// CHECK:  }

// -----

// CHECK-LABEL:  func.func @large_initial_value
// CHECK:         %[[CST:.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<65>
// CHECK:         iter_args({{.+}}, %arg2 = %[[CST]],{{.+}})
func.func @large_initial_value() -> (i1) {
  %bmc = verif.bmc bound 1 num_regs 1 initial_values [-1 : i65]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    verif.yield %clk: !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i65):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i65
  }
  func.return %bmc : i1
}

// -----

// CHECK-LABEL: func @test_refines_noreturn

// CHECK:     smt.solver() : () -> () {
// CHECK:       [[V0:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:       [[C1OUT:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<32>
// CHECK:       [[C2OUT:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<32>
// CHECK:       [[EQ1:%.+]] = smt.eq [[C1OUT]], [[V0]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ1]]
// CHECK:       [[EQ2:%.+]] = smt.eq [[C2OUT]], [[V0]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ2]]
// CHECK:       [[DIST:%.+]] = smt.distinct [[C1OUT]], [[C2OUT]] : !smt.bv<32>
// CHECK:       smt.assert [[DIST]]
// CHECK:       smt.check sat {
// CHECK-NEXT:   } unknown {
// CHECK-NEXT:   } unsat {
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @test_refines_noreturn() -> () {
  verif.refines first {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  } second {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  }

  // Trivial refines now also generates an smt.solver with false assertion
  // CHECK:       smt.solver
  // CHECK:       smt.constant false
  // CHECK:       smt.assert
  // CHECK:       smt.check sat
  // CHECK:     return
  verif.refines first {
  ^bb0():
    verif.yield
  } second {
  ^bb0():
    verif.yield
  }

  return
}

// -----

// CHECK-LABEL: func.func @test_refines_withreturn

// CHECK:     [[RT0:%.+]] = smt.solver() : () -> i1 {
// CHECK-DAG:   [[TRUE:%.+]]  = arith.constant true
// CHECK-DAG:   [[FALSE:%.+]] = arith.constant false
// CHECK:       [[V0:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:       [[C1OUT:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<32>
// CHECK:       [[C2OUT:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<32>
// CHECK:       [[EQ1:%.+]] = smt.eq [[C1OUT]], [[V0]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ1]]
// CHECK:       [[EQ2:%.+]] = smt.eq [[C2OUT]], [[V0]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ2]]
// CHECK:       [[DIST:%.+]] = smt.distinct [[C1OUT]], [[C2OUT]] : !smt.bv<32>
// CHECK:       smt.assert [[DIST]]
// CHECK:       [[V2:%.+]] = smt.check sat {
// CHECK-NEXT:     smt.yield [[FALSE]]
// CHECK-NEXT:   } unknown {
// CHECK-NEXT:     smt.yield [[FALSE]]
// CHECK-NEXT:   } unsat {
// CHECK-NEXT:     smt.yield [[TRUE]]
// CHECK-NEXT:   }
// CHECK-NEXT:   smt.yield [[V2]]
// CHECK-NEXT: }
// CHECK: return [[RT0]] : i1

func.func @test_refines_withreturn() -> i1 {
  %0 = verif.refines : i1 first {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  } second {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  }
  return %0 : i1
}

// -----

// CHECK-LABEL: func.func @test_refines_trivialreturn

// Trivial refines with return now generates an smt.solver with false assertion
// CHECK: [[SOLVER:%.+]] = smt.solver
// CHECK:   smt.constant false
// CHECK:   smt.assert
// CHECK:   smt.check sat
// CHECK: return [[SOLVER]] : i1

func.func @test_refines_trivialreturn() -> i1 {
  %0 = verif.refines : i1 first {
  ^bb0():
    verif.yield
  } second {
  ^bb0():
    verif.yield
  }
  return %0 : i1
}

// -----

// Source circuit non-deterministic

// CHECK-LABEL: func.func @nondet_to_det

// CHECK:     smt.solver()
// CHECK:       [[BVCST:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
// CHECK:       [[ALLQ:%.+]] = smt.forall {
// CHECK-NEXT:  ^bb0([[BVAR:%.+]]: !smt.bv<32>)
// CHECK-NEXT:    [[V0:%.+]] = smt.distinct [[BVAR]], [[BVCST]] : !smt.bv<32>
// CHECK-NEXT:    smt.yield [[V0]] : !smt.bool
// CHECK-NEXT:  }
// CHECK-NEXT:  smt.assert [[ALLQ]]
// CHECK-NEXT:  smt.check


func.func @nondet_to_det() -> () {
  verif.refines first {
  ^bb0():
    %nondet = smt.declare_fun : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %nondet : !smt.bv<32> to i32
    verif.yield %cc : i32
  } second {
  ^bb0():
    %const = smt.bv.constant #smt.bv<0> : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %const : !smt.bv<32> to i32
    verif.yield %cc : i32
  }
  return
}

// -----

// Target circuit non-deterministic

// CHECK-LABEL: func.func @det_to_nondet

// CHECK:     smt.solver()
// CHECK-DAG:   [[BVCST:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
// CHECK-DAG:   [[FREEVAR:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK-DAG:   [[C1OUT:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<32>
// CHECK-DAG:   [[C2OUT:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<32>
// CHECK:       [[EQ1:%.+]] = smt.eq [[C1OUT]], [[BVCST]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ1]]
// CHECK:       [[EQ2:%.+]] = smt.eq [[C2OUT]], [[FREEVAR]] : !smt.bv<32>
// CHECK:       smt.assert [[EQ2]]
// CHECK:       [[DIST:%.+]] = smt.distinct [[C1OUT]], [[C2OUT]]
// CHECK:       smt.assert [[DIST]]
// CHECK:       smt.check

func.func @det_to_nondet() -> () {
  verif.refines first {
  ^bb0():
    %const = smt.bv.constant #smt.bv<0> : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %const : !smt.bv<32> to i32
    verif.yield %cc : i32
  } second {
  ^bb0():
    %nondet = smt.declare_fun : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %nondet : !smt.bv<32> to i32
    verif.yield %cc : i32
  }
  return
}
// -----

// Both circuits non-deterministic

// CHECK-LABEL: func.func @nondet_to_nondet

// CHECK:     smt.solver()
// CHECK:       [[FREEVAR:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:       [[ALLQ:%.+]] = smt.forall {
// CHECK-NEXT:  ^bb0([[BVAR:%.+]]: !smt.bv<32>)
// CHECK-NEXT:    [[V0:%.+]] = smt.distinct [[BVAR]], [[FREEVAR]] : !smt.bv<32>
// CHECK-NEXT:    smt.yield [[V0]] : !smt.bool
// CHECK-NEXT:  }
// CHECK-NEXT:  smt.assert [[ALLQ]]
// CHECK-NEXT:  smt.check

func.func @nondet_to_nondet() -> () {
  verif.refines first {
  ^bb0():
    %nondet = smt.declare_fun : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %nondet : !smt.bv<32> to i32
    verif.yield %cc : i32
  } second {
  ^bb0():
    %nondet = smt.declare_fun : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %nondet : !smt.bv<32> to i32
    verif.yield %cc : i32
  }
  return
}

// -----

// Multiple non-deterministic values in the source circuit

// CHECK-LABEL: func.func @multi_nondet
// CHECK:     smt.solver()
// CHECK:       [[FREEVAR:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:       [[ALLQ:%.+]] = smt.forall {
// CHECK-NEXT:  ^bb0([[BVAR0:%.+]]: !smt.bv<32>, [[BVAR1:%.+]]: !smt.bv<32>)
// CHECK-DAG:     [[V0:%.+]] = smt.distinct [[BVAR0]], [[FREEVAR]] : !smt.bv<32>
// CHECK-DAG:     [[V1:%.+]] = smt.distinct [[BVAR1]], [[FREEVAR]] : !smt.bv<32>
// CHECK:         [[V2:%.+]] = smt.or [[V0]], [[V1]]
// CHECK-NEXT:    smt.yield [[V2]]
// CHECK-NEXT:  }
// CHECK-NEXT:  smt.assert [[ALLQ]]
// CHECK-NEXT:  smt.check

func.func @multi_nondet() -> () {
  verif.refines first {
  ^bb0():
    %nondet0 = smt.declare_fun : !smt.bv<32>
    %cc0 = builtin.unrealized_conversion_cast %nondet0 : !smt.bv<32> to i32
    %nondet1 = smt.declare_fun : !smt.bv<32>
    %cc1 = builtin.unrealized_conversion_cast %nondet1 : !smt.bv<32> to i32
    verif.yield %cc0, %cc1 : i32, i32
  } second {
  ^bb0():
    %nondet = smt.declare_fun : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %nondet : !smt.bv<32> to i32
    verif.yield %cc, %cc : i32, i32
  }
  return
}
