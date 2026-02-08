// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bin_op = "eq"
// SMTLIB-DAG: signal_lhs_cast_width = 2 : i32
// SMTLIB-DAG: signal_lhs_cast_signed = false
// SMTLIB-DAG: signal_lhs_cast_four_state = false
// SMTLIB-DAG: iff_bin_op = "ne"
// SMTLIB-DAG: iff_lhs_cast_width = 2 : i32
// SMTLIB-DAG: iff_lhs_cast_signed
// SMTLIB-DAG: iff_lhs_cast_four_state
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB-DAG: smt.bv.concat
// SMTLIB-DAG: smt.bv.repeat 1 times
// SMTLIB-DAG: smt.eq {{.*}} : !smt.bv<2>
// SMTLIB-DAG: smt.distinct {{.*}} : !smt.bv<2>

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bin_op = "eq"
// RUNTIME-DAG: signal_lhs_cast_width = 2 : i32
// RUNTIME-DAG: signal_lhs_cast_signed = false
// RUNTIME-DAG: signal_lhs_cast_four_state = false
// RUNTIME-DAG: iff_bin_op = "ne"
// RUNTIME-DAG: iff_lhs_cast_width = 2 : i32
// RUNTIME-DAG: iff_lhs_cast_signed
// RUNTIME-DAG: iff_lhs_cast_four_state
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME-DAG: smt.bv.concat
// RUNTIME-DAG: smt.bv.repeat 1 times
// RUNTIME-DAG: smt.eq {{.*}} : !smt.bv<2>
// RUNTIME-DAG: smt.distinct {{.*}} : !smt.bv<2>
//
// SMTLIB-LABEL: func.func @bmc_event_arm_witness_cast_two_state_projection
// SMTLIB-DAG: signal_lhs_cast_four_state = false
// SMTLIB: {{%.*}} = smt.bv.extract {{%.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// SMTLIB: {{%.*}} = smt.bv.extract {{%.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// SMTLIB: {{%.*}} = smt.bv.not {{%.*}} : !smt.bv<2>
// SMTLIB: {{%.*}} = smt.bv.and {{%.*}}, {{%.*}} : !smt.bv<2>
//
// RUNTIME-LABEL: func.func @bmc_event_arm_witness_cast_two_state_projection
// RUNTIME-DAG: signal_lhs_cast_four_state = false
// RUNTIME: {{%.*}} = smt.bv.extract {{%.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// RUNTIME: {{%.*}} = smt.bv.extract {{%.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// RUNTIME: {{%.*}} = smt.bv.not {{%.*}} : !smt.bv<2>
// RUNTIME: {{%.*}} = smt.bv.and {{%.*}}, {{%.*}} : !smt.bv<2>
//
// SMTLIB-LABEL: func.func @bmc_event_arm_witness_cast_slice_projection
// SMTLIB-DAG: signal_lhs_arg_lsb = 1 : i32
// SMTLIB-DAG: signal_lhs_arg_msb = 1 : i32
// SMTLIB: {{%.*}} = smt.bv.extract {{%.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// SMTLIB: {{%.*}} = smt.bv.extract {{%.*}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
//
// RUNTIME-LABEL: func.func @bmc_event_arm_witness_cast_slice_projection
// RUNTIME-DAG: signal_lhs_arg_lsb = 1 : i32
// RUNTIME-DAG: signal_lhs_arg_msb = 1 : i32
// RUNTIME: {{%.*}} = smt.bv.extract {{%.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// RUNTIME: {{%.*}} = smt.bv.extract {{%.*}} from 1 : (!smt.bv<2>) -> !smt.bv<1>

func.func @bmc_event_arm_witness_cast_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["a", "b", "ab", "mask", "ok"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bin_op = "ne", iff_lhs_arg_name = "b", iff_lhs_cast_four_state, iff_lhs_cast_signed, iff_lhs_cast_width = 2 : i32, iff_rhs_name = "mask", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "eq", signal_index = 0 : i32, signal_lhs_arg_name = "a", signal_lhs_cast_four_state = false, signal_lhs_cast_signed = false, signal_lhs_cast_width = 2 : i32, signal_rhs_name = "ab"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%a: i1, %b: i1, %ab: i2, %mask: i2, %ok: i1):
    verif.assert %ok : i1
    verif.yield
  }
  return
}

func.func @bmc_event_arm_witness_cast_two_state_projection() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["fs", "ref", "ok"],
    bmc_event_sources = [["signal[0]:both"]],
    bmc_event_source_details = [[{edge = "both", kind = "signal", label = "signal[0]:both", signal_bin_op = "eq", signal_index = 0 : i32, signal_lhs_arg_name = "fs", signal_lhs_cast_four_state = false, signal_lhs_cast_signed = false, signal_lhs_cast_width = 2 : i32, signal_rhs_name = "ref"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%fs: !hw.struct<value: i2, unknown: i2>, %ref: i2, %ok: i1):
    verif.assert %ok : i1
    verif.yield
  }
  return
}

func.func @bmc_event_arm_witness_cast_slice_projection() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["fs", "ref1", "ok"],
    bmc_event_sources = [["signal[0]:both"]],
    bmc_event_source_details = [[{edge = "both", kind = "signal", label = "signal[0]:both", signal_bin_op = "eq", signal_index = 0 : i32, signal_lhs_arg_lsb = 1 : i32, signal_lhs_arg_msb = 1 : i32, signal_lhs_arg_name = "fs", signal_lhs_cast_four_state = false, signal_lhs_cast_signed = false, signal_lhs_cast_width = 1 : i32, signal_rhs_name = "ref1"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%fs: !hw.struct<value: i2, unknown: i2>, %ref1: i1, %ok: i1):
    verif.assert %ok : i1
    verif.yield
  }
  return
}
