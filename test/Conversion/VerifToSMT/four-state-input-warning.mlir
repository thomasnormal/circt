// RUN: circt-opt %s --convert-verif-to-smt --split-input-file --verify-diagnostics -allow-unregistered-dialect

hw.type_scope @types {
  hw.typedecl @logic1_4s : !hw.struct<value: i1, unknown: i1>
}

func.func @bmc_four_state_warning() -> i1 {
  // expected-warning @below {{4-state inputs are unconstrained; consider --assume-known-inputs or full X-propagation support}}
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: !hw.struct<value: i1, unknown: i1>):
    %val = hw.struct_extract %sig["value"] : !hw.struct<value: i1, unknown: i1>
    verif.assert %val : i1
    verif.yield %sig : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}

// -----

func.func @lec_four_state_warning() -> i1 {
  // expected-warning @below {{4-state inputs are unconstrained; consider --assume-known-inputs or full X-propagation support}}
  %0 = verif.lec : i1 first {
  ^bb0(%arg0: !hw.struct<value: i1, unknown: i1>):
    verif.yield %arg0 : !hw.struct<value: i1, unknown: i1>
  } second {
  ^bb0(%arg0: !hw.struct<value: i1, unknown: i1>):
    verif.yield %arg0 : !hw.struct<value: i1, unknown: i1>
  }
  return %0 : i1
}

// -----

func.func @bmc_four_state_warning_alias() -> i1 {
  // expected-warning @below {{4-state inputs are unconstrained; consider --assume-known-inputs or full X-propagation support}}
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>):
    %val = hw.struct_extract %sig["value"] : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
    verif.assert %val : i1
    verif.yield %sig : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
  }
  func.return %bmc : i1
}

// -----

func.func @lec_four_state_warning_alias() -> i1 {
  // expected-warning @below {{4-state inputs are unconstrained; consider --assume-known-inputs or full X-propagation support}}
  %0 = verif.lec : i1 first {
  ^bb0(%arg0: !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>):
    verif.yield %arg0 : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
  } second {
  ^bb0(%arg0: !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>):
    verif.yield %arg0 : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
  }
  return %0 : i1
}
