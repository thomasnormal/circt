// RUN: circt-sim-compile %s --emit-snapshot -o %t.csnap 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %t.csnap 2>&1 | FileCheck %s --check-prefix=SNAP-ON
// RUN: env CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=0 circt-sim %t.csnap 2>&1 | FileCheck %s --check-prefix=SNAP-OFF

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] Native module init functions: 1
// COMPILE: [circt-sim-compile] Wrote snapshot to
//
// SNAP-ON: [circt-sim] Loading snapshot from
// SNAP-ON: [circt-sim] Snapshot auto-enabled native module init
// SNAP-ON: [circt-sim] Native module init: top
// SNAP-ON: out=123
//
// SNAP-OFF: [circt-sim] Loading snapshot from
// SNAP-OFF-NOT: [circt-sim] Snapshot auto-enabled native module init
// SNAP-OFF-NOT: [circt-sim] Native module init: top
// SNAP-OFF: out=123

llvm.mlir.global internal @g_counter(0 : i32) : i32

func.func @read_counter() -> i32 {
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %v = llvm.load %ptr : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %c123 = llvm.mlir.constant(123 : i32) : i32
  llvm.store %c123, %ptr : i32, !llvm.ptr

  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_counter() : () -> i32
    %fmtV = sim.fmt.dec %r signed : i32
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %fmtOut
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
