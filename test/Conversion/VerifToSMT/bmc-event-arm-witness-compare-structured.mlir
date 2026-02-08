// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bin_op = "lt"
// SMTLIB-DAG: signal_cmp_signed = true
// SMTLIB-DAG: iff_bin_op = "ge"
// SMTLIB-DAG: iff_cmp_signed = false
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.bv.cmp slt {{.*}} : !smt.bv<3>
// SMTLIB: smt.bv.cmp uge {{.*}} : !smt.bv<3>

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bin_op = "lt"
// RUNTIME-DAG: signal_cmp_signed = true
// RUNTIME-DAG: iff_bin_op = "ge"
// RUNTIME-DAG: iff_cmp_signed = false
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.bv.cmp slt {{.*}} : !smt.bv<3>
// RUNTIME: smt.bv.cmp uge {{.*}} : !smt.bv<3>

func.func @bmc_event_arm_witness_compare_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["sbus", "smask", "bus", "mask", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bin_op = "ge", iff_cmp_signed = false, iff_lhs_name = "bus", iff_rhs_name = "mask", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "lt", signal_cmp_signed = true, signal_index = 0 : i32, signal_lhs_name = "sbus", signal_rhs_name = "smask"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%sbus: i3, %smask: i3, %bus: i3, %mask: i3, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
