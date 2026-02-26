// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s

// CHECK: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// CHECK: [circt-sim-compile] Stripped 1 functions with non-LLVM ops
// CHECK: [circt-sim-compile] Top residual non-LLVM strip reasons:
// CHECK: 1x body_nonllvm_op:builtin.unrealized_conversion_cast

module {
  func.func private @ok() -> i32 {
    %c7_i32 = hw.constant 7 : i32
    return %c7_i32 : i32
  }

  // This function is accepted by the front-end compilability filter
  // (pointer->function cast and call_indirect), but lowering leaves a residual
  // non-LLVM cast because the indirect signature uses !hw.struct.
  func.func private @strip_me(%fnptr: !llvm.ptr,
                              %argraw: !llvm.struct<(i8)>) -> i32 {
    %arg = builtin.unrealized_conversion_cast %argraw : !llvm.struct<(i8)> to !hw.struct<f: i8>
    %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to (!hw.struct<f: i8>) -> i32
    %r = func.call_indirect %fn(%arg) : (!hw.struct<f: i8>) -> i32
    return %r : i32
  }
}
