// RUN: circt-sim-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: when a non-compiled llvm.func with a `personality` attribute is
// externalized and given an interpreter trampoline body, MLIR->LLVM IR
// translation must not crash.
//
// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
// COMPILE: [circt-sim-compile] Wrote LLVM IR to

func.func @entry(%x: i32) -> i32 {
  %r = llvm.call @invoke_wrap(%x) : (i32) -> i32
  return %r : i32
}

llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @callee(i32) -> i32 {
^bb0(%x: i32):
  llvm.return %x : i32
}

llvm.func @invoke_wrap(%x: i32) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %tag = llvm.mlir.constant(7 : i32) : i32
  %res = llvm.invoke @callee(%x) to ^bb1 unwind ^bb2(%tag : i32) : (i32) -> i32
^bb1:
  llvm.return %res : i32
^bb2(%u: i32):
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
}
