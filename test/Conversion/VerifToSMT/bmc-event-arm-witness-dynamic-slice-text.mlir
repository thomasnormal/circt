// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_expr = "bus[(idx - 1)]"
// SMTLIB-DAG: signal_expr = "bus[(jdx + 1) +: 2]"
// SMTLIB-DAG: iff_expr = "bus[(idx - 1) -: 1]"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// SMTLIB: smt.bv.concat
// SMTLIB: smt.bv.add
// SMTLIB: smt.bv.lshr
// SMTLIB: from 0 : (!smt.bv<8>) -> !smt.bv<2>

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_expr = "bus[(idx - 1)]"
// RUNTIME-DAG: signal_expr = "bus[(jdx + 1) +: 2]"
// RUNTIME-DAG: iff_expr = "bus[(idx - 1) -: 1]"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// RUNTIME: smt.bv.concat
// RUNTIME: smt.bv.add
// RUNTIME: smt.bv.lshr
// RUNTIME: from 0 : (!smt.bv<8>) -> !smt.bv<2>

func.func @bmc_event_arm_witness_dynamic_slice_text() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "idx", "jdx", "en"],
    bmc_event_sources = [["signal[0]:both:iff_expr", "signal[1]:both:iff_expr"]],
    bmc_event_source_details = [[{edge = "both", iff_expr = "en", kind = "signal", label = "signal[0]:both:iff_expr", signal_expr = "bus[(idx - 1)]", signal_index = 0 : i32}, {edge = "both", iff_expr = "bus[(idx - 1) -: 1]", kind = "signal", label = "signal[1]:both:iff_expr", signal_expr = "bus[(jdx + 1) +: 2]", signal_index = 1 : i32}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%bus: i8, %idx: i3, %jdx: i3, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
