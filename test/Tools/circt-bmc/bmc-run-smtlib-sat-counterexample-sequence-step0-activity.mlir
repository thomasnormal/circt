// RUN: circt-bmc -b 1 --run-smtlib --z3-path=%S/Inputs/fake-z3-sat-model-sequence-step0.sh --print-counterexample --module top %s 2>&1 | FileCheck %s

hw.module @top(in %seq: i1, out out: i1) attributes {initial_values = [], moore.event_source_details = [[{kind = "sequence", label = "sequence[0]", sequence_index = 0 : i32, sequence_name = "seq"}]], moore.event_sources = [["sequence[0]"]], num_regs = 0 : i32} {
  verif.assert %seq : i1
  hw.output %seq : i1
}

// CHECK-DAG: sat
// CHECK-DAG: BMC_RESULT=SAT
// CHECK-DAG: mixed event sources:
// CHECK-DAG: [0] sequence[0]
// CHECK-DAG: estimated event-arm activity:
// CHECK-DAG: [0][0] sequence[0] -> step 0
// CHECK-DAG: estimated fired arms by step:
// CHECK-DAG: [0] step 0 -> sequence[0]
