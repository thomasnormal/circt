// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: tx_write_packet_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @tx_write_packet(
      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr,
      %arg3: !llhd.ref<!llvm.ptr>) {
    func.return
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "tx_write_packet_ok\0A"
    %null = llvm.mlir.zero : !llvm.ptr
    %one = hw.constant 1 : i64

    llhd.process {
      %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %slot_ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!llvm.ptr>
      func.call @tx_write_packet(%null, %null, %null, %slot_ref) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llhd.ref<!llvm.ptr>) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
