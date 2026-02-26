// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
//
// When no FuncId entry table is emitted (e.g. no process/module function IDs),
// tagged-indirect lowering must not synthesize references to
// @__circt_sim_func_entries, otherwise linking fails.
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] Stripped 1 functions with non-LLVM ops
// COMPILE: [circt-sim-compile] Top residual non-LLVM strip reasons:
// COMPILE: 1x sig_nonllvm_arg:!hw.struct<f: i8>
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
// COMPILE-NOT: LowerTaggedIndirectCalls: lowered
// COMPILE-NOT: Linking failed

module {
  // Intentionally non-LLVM signature so this gets stripped and does not
  // contribute to FuncId entry table emission.
  func.func private @callee(%arg: !hw.struct<f: i8>) -> i32 {
    %x = hw.struct_extract %arg["f"] : !hw.struct<f: i8>
    %x32 = arith.extui %x : i8 to i32
    return %x32 : i32
  }

  // LLVM-compatible signature with call_indirect from ptr->func cast.
  // Prior to the fix this path lowered to @__circt_sim_func_entries even when
  // no table existed in the module, causing a link-time failure.
  func.func private @driver(%fnptr: !llvm.ptr,
                            %argraw: !llvm.struct<(i8)>) -> i32 {
    %arg = builtin.unrealized_conversion_cast %argraw : !llvm.struct<(i8)> to !hw.struct<f: i8>
    %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to (!hw.struct<f: i8>) -> i32
    %r = func.call_indirect %fn(%arg) : (!hw.struct<f: i8>) -> i32
    return %r : i32
  }
}
