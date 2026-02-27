// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=STATS

// Regression: region-based pointer global initializers that use addressof+gep
// must be materialized in interpreter global init.
//
// COMPILE: [circt-compile] Global patches: 1 mutable globals
// COMPILE: [circt-compile] Arena migration skipped 1 mutable globals with non-rewritable constant users
// COMPILE: [circt-compile] Arena: 0 globals, 0 bytes
//
// SIM: out=99
// COMPILED: out=99
//
// STATS-DAG: [circt-sim] arena_globals:                    0
// STATS-DAG: [circt-sim] arena_size_bytes:                 0
// STATS-DAG: [circt-sim] global_patch_count:               1
// STATS-DAG: [circt-sim] global_patch_bytes:               8
// STATS-DAG: [circt-sim] mutable_globals_total:            1
// STATS-DAG: [circt-sim] mutable_globals_arena:            0
// STATS-DAG: [circt-sim] mutable_globals_patch:            1
// STATS-DAG: [circt-sim] mutable_bytes_total:              8
// STATS-DAG: [circt-sim] mutable_bytes_arena:              0
// STATS-DAG: [circt-sim] mutable_bytes_patch:              8
// STATS: out=99

llvm.mlir.global internal @g_arr(0 : i64) : !llvm.array<2 x i32>
llvm.mlir.global internal constant @g_arr_p1() : !llvm.ptr {
  %base = llvm.mlir.addressof @g_arr : !llvm.ptr
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %p1 = llvm.getelementptr %base[%c0, %c1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i32>
  llvm.return %p1 : !llvm.ptr
}

func.func @write_p1(%v: i32) {
  %pp = llvm.mlir.addressof @g_arr_p1 : !llvm.ptr
  %p1 = llvm.load %pp : !llvm.ptr -> !llvm.ptr
  llvm.store %v, %p1 : i32, !llvm.ptr
  return
}

func.func @read_p1() -> i32 {
  %base = llvm.mlir.addressof @g_arr : !llvm.ptr
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %p1 = llvm.getelementptr %base[%c0, %c1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i32>
  %v = llvm.load %p1 : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %c99 = llvm.mlir.constant(99 : i32) : i32
  %t5 = hw.constant 5000000 : i64
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    func.call @write_p1(%c99) : (i32) -> ()
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t5
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_p1() : () -> i32
    %vf = sim.fmt.dec %r signed : i32
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
