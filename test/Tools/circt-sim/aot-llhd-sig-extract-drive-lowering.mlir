// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=8
//
// COMPILED: out=8

func.func @set_bit3_with_sig_extract() -> i8 {
  %one = llvm.mlir.constant(1 : i64) : i64
  %zero = hw.constant 0 : i8
  %true = hw.constant true
  %low = hw.constant 3 : i3
  %t0 = llhd.constant_time <0ns, 0d, 1e>

  %slot = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
  llvm.store %zero, %slot : i8, !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<i8>
  %bit = llhd.sig.extract %ref from %low : <i8> -> <i1>
  llhd.drv %bit, %true after %t0 : i1

  %out = llvm.load %slot : !llvm.ptr -> i8
  return %out : i8
}

hw.module @top() {
  llhd.process {
    %x = func.call @set_bit3_with_sig_extract() : () -> i8
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i8
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
