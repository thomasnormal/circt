// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_create_random_seed"(%type_id: !llvm.struct<(ptr, i64)>, %inst_id: !llvm.struct<(ptr, i64)>) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero64 = hw.constant 0 : i64
    %seed_fmt_prefix = sim.fmt.literal "seed="
    %nl = sim.fmt.literal "\0A"
    %packed_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %packed_ptr = llvm.insertvalue %null, %packed_undef[0] : !llvm.struct<(ptr, i64)>
    %packed_str = llvm.insertvalue %zero64, %packed_ptr[1] : !llvm.struct<(ptr, i64)>

    llhd.process {
      %seed = func.call @"uvm_pkg::uvm_create_random_seed"(%packed_str, %packed_str) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> i32
      %seed_fmt = sim.fmt.dec %seed : i32
      %out = sim.fmt.concat (%seed_fmt_prefix, %seed_fmt, %nl)
      sim.proc.print %out
      llhd.halt
    }

    hw.output
  }
}
