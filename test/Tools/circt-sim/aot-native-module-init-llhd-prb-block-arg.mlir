// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=NATIVE

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Native module init functions: 1
// COMPILE: [circt-compile] Native module init modules: 1 emitted / 1 total
//
// SIM: done
//
// NATIVE: [circt-sim] Native module init: top
// NATIVE: done

llvm.mlir.global internal @g_probe_ptr(0 : i64) : i64

func.func @read_probe_ptr() -> i64 {
  %ptr = llvm.mlir.addressof @g_probe_ptr : !llvm.ptr
  %v = llvm.load %ptr : !llvm.ptr -> i64
  return %v : i64
}

hw.module @top(in %in_ref : !llhd.ref<!llvm.ptr>) {
  %probed_ptr = llhd.prb %in_ref : !llvm.ptr
  %as_i64 = builtin.unrealized_conversion_cast %probed_ptr : !llvm.ptr to i64
  %slot = llvm.mlir.addressof @g_probe_ptr : !llvm.ptr
  llvm.store %as_i64, %slot : i64, !llvm.ptr

  %fmtPrefix = sim.fmt.literal "done"
  %fmtNl = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %ignore = func.call @read_probe_ptr() : () -> i64
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtNl)
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
