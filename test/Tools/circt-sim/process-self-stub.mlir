// RUN: circt-sim %s --top TestProcessSelfStub 2>&1 | FileCheck %s
// Tests that the old stub @self() function (from earlier compilations that
// returned null) is intercepted by the interpreter to return a valid
// process handle. This fixes UVM's "run_test() invoked from a non process
// context" error.

module {
  // Old-style stub that always returns null - the interpreter should
  // intercept this and return a valid process handle instead.
  func.func private @self() -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    return %0 : !llvm.ptr
  }

  hw.module @TestProcessSelfStub() {
    %null = llvm.mlir.zero : !llvm.ptr
    %time = llhd.constant_time <0ns, 0d, 1e>
    %yes = sim.fmt.literal "process context: YES\0A"
    %no = sim.fmt.literal "process context: NO\0A"

    llhd.process {
      // Call the old-style @self() stub - interpreter should intercept
      // and return non-null process handle
      %p = func.call @self() : () -> !llvm.ptr

      // Check if result is non-null (valid process context)
      %is_null = llvm.icmp "eq" %p, %null : !llvm.ptr
      %msg = arith.select %is_null, %no, %yes : !sim.fstring

      // CHECK: process context: YES
      sim.proc.print %msg
      llhd.halt
    }
    hw.output
  }
}
