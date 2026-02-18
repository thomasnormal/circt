// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: set_config_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @"UartTxSequencePkg::UartTxTransmitterSequence::setConfig"(
      %self: !llvm.ptr, %cfg: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt = sim.fmt.literal "set_config_ok\0A"

    llhd.process {
      func.call @"UartTxSequencePkg::UartTxTransmitterSequence::setConfig"(
          %null, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
