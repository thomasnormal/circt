// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: {{[0-9]+}} total, {{[0-9]+}} external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// SIM: out=42
//
// COMPILED: out=42

func.func private @read_ref(%arg0: !llhd.ref<i32>) -> i32 {
  %ptr = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<i32> to !llvm.ptr
  %value = llvm.load %ptr : !llvm.ptr -> i32
  return %value : i32
}

func.func private @read_via_ref_abi(%ptr: !llvm.ptr) -> i32 {
  %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i32>
  %value = func.call @read_ref(%ref) : (!llhd.ref<i32>) -> i32
  return %value : i32
}

hw.module @top() {
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c42_i32 = hw.constant 42 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %c42_i32 : i32
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"

  llhd.process {
    %slot = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr
    llvm.store %c42_i32, %slot : i32, !llvm.ptr
    %v = func.call @read_via_ref_abi(%slot) : (!llvm.ptr) -> i32
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^do_print(%v : i32)
  ^do_print(%x: i32):
    llhd.drv %sig, %x after %t0 : i32
    %p = llhd.prb %sig : i32
    %fmtV = sim.fmt.dec %p signed : i32
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
