// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen

llvm.func @callee_llvm(%pair: !llvm.struct<(i64, i64)>) -> i32 {
  %a = llvm.extractvalue %pair[0] : !llvm.struct<(i64, i64)>
  %b = llvm.extractvalue %pair[1] : !llvm.struct<(i64, i64)>
  %a32 = llvm.trunc %a : i64 to i32
  %b32 = llvm.trunc %b : i64 to i32
  %sum = llvm.add %a32, %b32 : i32
  llvm.return %sum : i32
}

func.func @driver() -> i32 {
  %v = hw.constant 7 : i64
  %u = hw.constant 5 : i64
  %pair = hw.struct_create (%v, %u) : !hw.struct<value: i64, unknown: i64>
  %callee_ptr = llvm.mlir.addressof @callee_llvm : !llvm.ptr
  %fn = builtin.unrealized_conversion_cast %callee_ptr : !llvm.ptr to (!hw.struct<value: i64, unknown: i64>) -> i32
  %r = func.call_indirect %fn(%pair) : (!hw.struct<value: i64, unknown: i64>) -> i32
  return %r : i32
}
