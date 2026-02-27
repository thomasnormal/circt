// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: fold 4-state hw.struct_create/hw.struct_extract and
// hw.aggregate_constant/hw.struct_extract in AOT function lowering.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// SIM: out=47
//
// COMPILED: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// COMPILED: out=47

func.func @from_create(%x: i32) -> i32 {
  %u = arith.constant 0 : i32
  %s = hw.struct_create (%x, %u) : !hw.struct<value: i32, unknown: i32>
  %v = hw.struct_extract %s["value"] : !hw.struct<value: i32, unknown: i32>
  return %v : i32
}

func.func @from_agg() -> i32 {
  %s = hw.aggregate_constant [42 : i32, 0 : i32] : !hw.struct<value: i32, unknown: i32>
  %v = hw.struct_extract %s["value"] : !hw.struct<value: i32, unknown: i32>
  return %v : i32
}

hw.module @test() {
  %c5_i32 = hw.constant 5 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %a = func.call @from_create(%c5_i32) : (i32) -> i32
    %b = func.call @from_agg() : () -> i32
    %sum = arith.addi %a, %b : i32
    %fmt_val = sim.fmt.dec %sum : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c15_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
