// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: get_minimum_transactions_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @"axi4_slave_pkg::axi4_slave_agent_config::get_minimum_transactions"(
      %arg0: !llvm.ptr) -> i32 {
    %value = hw.constant 3 : i32
    return %value : i32
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "get_minimum_transactions_ok\0A"
    %null = llvm.mlir.zero : !llvm.ptr

    llhd.process {
      %result = func.call @"axi4_slave_pkg::axi4_slave_agent_config::get_minimum_transactions"(%null) : (!llvm.ptr) -> i32
      %zero = hw.constant 0 : i32
      %sum = comb.add %result, %zero : i32
      %unused = comb.icmp uge %sum, %zero : i32
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
