// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: bmc_event_source_details =
// SMTLIB-DAG: signal_dyn_index_name = "idx"
// SMTLIB-DAG: signal_dyn_sign = -1 : i32
// SMTLIB-DAG: signal_dyn_offset = 7 : i32
// SMTLIB-DAG: signal_dyn_width = 2 : i32
// SMTLIB-DAG: iff_dyn_index_name = "jdx"
// SMTLIB-DAG: iff_dyn_sign = 1 : i32
// SMTLIB-DAG: iff_dyn_offset = 0 : i32
// SMTLIB-DAG: iff_dyn_width = 1 : i32
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.bv.concat
// SMTLIB: smt.bv.neg
// SMTLIB: smt.bv.add
// SMTLIB: smt.bv.lshr
// SMTLIB: from 0 : (!smt.bv<8>) -> !smt.bv<2>

// RUNTIME: smt.solver
// RUNTIME-DAG: bmc_event_source_details =
// RUNTIME-DAG: signal_dyn_index_name = "idx"
// RUNTIME-DAG: signal_dyn_sign = -1 : i32
// RUNTIME-DAG: signal_dyn_offset = 7 : i32
// RUNTIME-DAG: signal_dyn_width = 2 : i32
// RUNTIME-DAG: iff_dyn_index_name = "jdx"
// RUNTIME-DAG: iff_dyn_sign = 1 : i32
// RUNTIME-DAG: iff_dyn_offset = 0 : i32
// RUNTIME-DAG: iff_dyn_width = 1 : i32
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.bv.concat
// RUNTIME: smt.bv.neg
// RUNTIME: smt.bv.add
// RUNTIME: smt.bv.lshr
// RUNTIME: from 0 : (!smt.bv<8>) -> !smt.bv<2>

func.func @bmc_event_arm_witness_dynamic_slice_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "idx", "jdx", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_dyn_index_name = "jdx", iff_dyn_offset = 0 : i32, iff_dyn_sign = 1 : i32, iff_dyn_width = 1 : i32, iff_name = "bus", kind = "signal", label = "signal[0]:both:iff", signal_dyn_index_name = "idx", signal_dyn_offset = 7 : i32, signal_dyn_sign = -1 : i32, signal_dyn_width = 2 : i32, signal_index = 0 : i32, signal_name = "bus"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%bus: i8, %idx: i3, %jdx: i2, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
