// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=STATS

// Regression: region initializers must resolve addressof references even when
// the target global is declared later in symbol order.
//
// COMPILE: [circt-compile] Global patches: 1 mutable globals
// COMPILE: [circt-compile] Arena migration skipped 1 mutable globals with non-rewritable constant users
// COMPILE: [circt-compile] Arena: 0 globals, 0 bytes
//
// SIM: out=55
// COMPILED: out=55
//
// STATS-DAG: [circt-sim] arena_globals:                    0
// STATS-DAG: [circt-sim] arena_size_bytes:                 0
// STATS-DAG: [circt-sim] global_patch_count:               1
// STATS-DAG: [circt-sim] mutable_globals_total:            1
// STATS-DAG: [circt-sim] mutable_globals_arena:            0
// STATS-DAG: [circt-sim] mutable_globals_patch:            1
// STATS: out=55

llvm.mlir.global internal constant @g_late_ptr() : !llvm.ptr {
  %p = llvm.mlir.addressof @g_late : !llvm.ptr
  llvm.return %p : !llvm.ptr
}

llvm.mlir.global internal @g_late(0 : i32) : i32

func.func @write_late_ptr(%v: i32) {
  %pp = llvm.mlir.addressof @g_late_ptr : !llvm.ptr
  %p = llvm.load %pp : !llvm.ptr -> !llvm.ptr
  llvm.store %v, %p : i32, !llvm.ptr
  return
}

func.func @read_late() -> i32 {
  %p = llvm.mlir.addressof @g_late : !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %c55 = llvm.mlir.constant(55 : i32) : i32
  %t5 = hw.constant 5000000 : i64
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    func.call @write_late_ptr(%c55) : (i32) -> ()
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t5
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_late() : () -> i32
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
