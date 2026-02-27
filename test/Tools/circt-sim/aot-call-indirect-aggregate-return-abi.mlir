// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --top test | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --top test --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: indirect calls with aggregate returns must be ABI-rewritten
// consistently with aggregate-return callee flattening (sret).
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] 2 functions + 0 processes ready for codegen
//
// SIM: sum=83
// COMPILED: sum=83

func.func private @"math::mk_pair"(%x: i64) -> !hw.struct<value: i64, unknown: i64> {
  %one = hw.constant 1 : i64
  %y = arith.addi %x, %one : i64
  %pair = hw.struct_create (%x, %y) : !hw.struct<value: i64, unknown: i64>
  return %pair : !hw.struct<value: i64, unknown: i64>
}

func.func @driver(%fptr: !llvm.ptr, %x: i64) -> i64 {
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64) -> !hw.struct<value: i64, unknown: i64>
  %pair = func.call_indirect %fn(%x) : (i64) -> !hw.struct<value: i64, unknown: i64>
  %a = hw.struct_extract %pair["value"] : !hw.struct<value: i64, unknown: i64>
  %b = hw.struct_extract %pair["unknown"] : !hw.struct<value: i64, unknown: i64>
  %sum = arith.addi %a, %b : i64
  return %sum : i64
}

llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::mk_pair"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmtPrefix = sim.fmt.literal "sum="
  %fmtNl = sim.fmt.literal "\0A"
  %c41 = hw.constant 41 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %vtable = llvm.mlir.addressof @"math::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %sum = func.call @driver(%fptr, %c41) : (!llvm.ptr, i64) -> i64

    %fmtV = sim.fmt.dec %sum : i64
    %msg = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %msg
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
