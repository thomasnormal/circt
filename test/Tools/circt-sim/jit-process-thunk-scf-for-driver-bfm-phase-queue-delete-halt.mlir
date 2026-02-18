// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: scf_for_driver_bfm_queue_delete_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @"axi4_slave_driver_bfm::axi4_write_address_phase"(
      %arg0: !llvm.ptr, %arg1: !llhd.ref<i1>) {
    func.return
  }

  llvm.func @__moore_queue_delete_index(!llvm.ptr, i32, i64)

  hw.module @top() {
    %fmt = sim.fmt.literal "scf_for_driver_bfm_queue_delete_ok\0A"
    %null = llvm.mlir.zero : !llvm.ptr
    %slotCount = hw.constant 1 : i64
    %idx = hw.constant 0 : i32
    %size = hw.constant 1 : i64
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index

    llhd.process {
      %slot = llvm.alloca %slotCount x i1 : (i64) -> !llvm.ptr
      %slot_ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<i1>
      scf.for %i = %lb to %ub step %step {
        func.call @"axi4_slave_driver_bfm::axi4_write_address_phase"(%null, %slot_ref) : (!llvm.ptr, !llhd.ref<i1>) -> ()
        llvm.call @__moore_queue_delete_index(%null, %idx, %size) : (!llvm.ptr, i32, i64) -> ()
      }
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
