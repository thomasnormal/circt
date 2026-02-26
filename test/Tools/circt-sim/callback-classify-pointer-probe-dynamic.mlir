// RUN: env CIRCT_SIM_COMPILE_REPORT=1 circt-sim %s --skip-passes --max-time=2000000 2>&1 | FileCheck %s

// Pointer-backed probe-only waits must not become CallbackStaticObserved.
// The pointer handle is stable while pointee fields may change; these waits
// require dynamic sensitivity handling.

// CHECK: ExecModel breakdown (registered processes):
// CHECK: CallbackDynamicWait: 1
// CHECK: CallbackTimeOnly: 1
// CHECK-NOT: CallbackStaticObserved: 1

func.func private @malloc(i64) -> !llvm.ptr

hw.module @test() {
  %c16_i64 = arith.constant 16 : i64
  %c2000000_i64 = hw.constant 2000000 : i64

  %ptr = func.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
  %sig = llhd.sig %ptr : !llvm.ptr

  llhd.process {
    cf.br ^wait
  ^wait:
    %p = llhd.prb %sig : !llvm.ptr
    llhd.wait ^body
  ^body:
    cf.br ^wait
  }

  llhd.process {
    %t = llhd.int_to_time %c2000000_i64
    llhd.wait delay %t, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
