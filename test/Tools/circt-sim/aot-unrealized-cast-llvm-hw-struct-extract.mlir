// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=42
//
// COMPILED: out=42

func.func @read_field() -> i32 {
  %u = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  %c11 = llvm.mlir.constant(11 : i32) : i32
  %c42 = llvm.mlir.constant(42 : i32) : i32
  %s0 = llvm.insertvalue %c11, %u[0] : !llvm.struct<(i32, i32)>
  %s1 = llvm.insertvalue %c42, %s0[1] : !llvm.struct<(i32, i32)>
  %h = builtin.unrealized_conversion_cast %s1 : !llvm.struct<(i32, i32)> to !hw.struct<a: i32, b: i32>
  %b = hw.struct_extract %h["b"] : !hw.struct<a: i32, b: i32>
  return %b : i32
}

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_field() : () -> i32
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
