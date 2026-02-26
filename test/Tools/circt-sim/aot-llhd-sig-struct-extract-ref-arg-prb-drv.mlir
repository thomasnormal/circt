// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=8
//
// COMPILED: out=8

func.func @bump_struct_arg(%ref: !llhd.ref<!hw.struct<a: i32, b: i64>>) -> i32 {
  %c1 = hw.constant 1 : i32
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %aRef = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: i32, b: i64>>
  %a = llhd.prb %aRef : i32
  %next = comb.add %a, %c1 : i32
  llhd.drv %aRef, %next after %t0 : i32
  %after = llhd.prb %aRef : i32
  return %after : i32
}

hw.module @top() {
  %one = llvm.mlir.constant(1 : i64) : i64
  %base = llvm.mlir.undef : !llvm.struct<(i32, i64)>
  %c7 = hw.constant 7 : i32
  %c9 = hw.constant 9 : i64

  llhd.process {
    %slot = llvm.alloca %one x !llvm.struct<(i32, i64)> : (i64) -> !llvm.ptr
    %v0 = llvm.insertvalue %c7, %base[0] : !llvm.struct<(i32, i64)>
    %v1 = llvm.insertvalue %c9, %v0[1] : !llvm.struct<(i32, i64)>
    llvm.store %v1, %slot : !llvm.struct<(i32, i64)>, !llvm.ptr
    %ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!hw.struct<a: i32, b: i64>>

    %x = func.call @bump_struct_arg(%ref) : (!llhd.ref<!hw.struct<a: i32, b: i64>>) -> i32

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
