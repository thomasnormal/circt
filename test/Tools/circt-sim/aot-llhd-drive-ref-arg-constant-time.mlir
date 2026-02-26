// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: {{[0-9]+}} total, {{[0-9]+}} external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] 2 functions + 0 processes ready for codegen
//
// SIM: out=42
//
// COMPILED: out=42

func.func private @set_ref_i32(%out: !llhd.ref<i32>, %v: i32) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  llhd.drv %out, %v after %t0 : i32
  return
}

func.func private @run_ref_drive() -> i32 {
  %one = llvm.mlir.constant(1 : i64) : i64
  %c42 = hw.constant 42 : i32
  %slot = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<i32>
  func.call @set_ref_i32(%ref, %c42) : (!llhd.ref<i32>, i32) -> ()
  %out = llvm.load %slot : !llvm.ptr -> i32
  return %out : i32
}

hw.module @top() {
  llhd.process {
    %x = func.call @run_ref_drive() : () -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
