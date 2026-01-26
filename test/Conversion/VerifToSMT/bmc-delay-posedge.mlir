// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @delay_posedge
// CHECK:       [[BVTRUE:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK:       scf.for
// CHECK:         [[LOOP:%.+]] = func.call @bmc_loop
// CHECK:         [[OLDCLOCKLOW:%.+]] = smt.bv.not
// CHECK:         [[BVPOSEDGE:%.+]] = smt.bv.and [[OLDCLOCKLOW]], [[LOOP]]
// CHECK:         [[ISPOSEDGE:%.+]] = smt.eq [[BVPOSEDGE]], [[BVTRUE]]
// CHECK:         [[BUFNEXT:%.+]] = smt.ite [[ISPOSEDGE]]
// CHECK-LABEL: func.func @bmc_circuit
// CHECK:       [[TRUE:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK:       smt.eq %arg{{[0-9]+}}, [[TRUE]]
// CHECK:       smt.eq %arg{{[0-9]+}}, [[TRUE]]

func.func @delay_posedge() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
    %c0 = hw.constant false
    %clk = seq.to_clock %c0
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from = seq.from_clock %clk
    %true = hw.constant true
    %n = comb.xor %from, %true : i1
    %nclk = seq.to_clock %n
    verif.yield %nclk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %del = ltl.delay %sig, 1, 0 : i1
    %prop = ltl.implication %sig, %del : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
