// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=STATS

// Regression: mutable globals referenced by non-rewritable constant users
// (for example a constant global pointer initializer) must stay in the legacy
// patch table. Arena migration cannot encode runtime arena-base-relative
// addresses inside constant initializers.
//
// COMPILE: [circt-compile] Global patches: 1 mutable globals
// COMPILE: [circt-compile] Arena migration skipped 1 mutable globals with non-rewritable constant users
// COMPILE: [circt-compile] Arena: 0 globals, 0 bytes
//
// SIM: out=77
//
// COMPILED: out=77
//
// STATS-DAG: [circt-sim] arena_globals:                    0
// STATS-DAG: [circt-sim] arena_size_bytes:                 0
// STATS-DAG: [circt-sim] global_patch_count:               1
// STATS-DAG: [circt-sim] mutable_globals_total:            1
// STATS-DAG: [circt-sim] mutable_globals_arena:            0
// STATS-DAG: [circt-sim] mutable_globals_patch:            1
// STATS: out=77

llvm.mlir.global internal @g_counter(0 : i32) : i32
llvm.mlir.global internal constant @g_counter_ptr() : !llvm.ptr {
  %p = llvm.mlir.addressof @g_counter : !llvm.ptr
  llvm.return %p : !llvm.ptr
}

func.func @write_through_const_ptr(%v: i32) {
  %gpAddr = llvm.mlir.addressof @g_counter_ptr : !llvm.ptr
  %gp = llvm.load %gpAddr : !llvm.ptr -> !llvm.ptr
  llvm.store %v, %gp : i32, !llvm.ptr
  return
}

func.func @read_counter() -> i32 {
  %gAddr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %x = llvm.load %gAddr : !llvm.ptr -> i32
  return %x : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %c77 = llvm.mlir.constant(77 : i32) : i32
  %t5 = hw.constant 5000000 : i64
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    func.call @write_through_const_ptr(%c77) : (i32) -> ()
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t5
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_counter() : () -> i32
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
