// RUN: circt-sim %s | FileCheck %s

// Test probing a signal reference passed as an LLVM function argument.
// This verifies that the interpreter correctly creates temporary signal mappings
// for function parameters when calling functions with signal references.

// CHECK: func_probe_result=42

module {
  // LLVM function that takes a pointer (representing signal ref) and probes it
  // The pointer is cast to !llhd.ref inside the function for probing
  llvm.func @probe_signal(%sig_ptr: !llvm.ptr) -> i8 {
    %ref = builtin.unrealized_conversion_cast %sig_ptr : !llvm.ptr to !llhd.ref<i8>
    %val = llhd.prb %ref : i8
    llvm.return %val : i8
  }

  hw.module @test() {
    %c1_i64 = hw.constant 1000000 : i64
    %c42_i8 = hw.constant 42 : i8
    %fmt_pre = sim.fmt.literal "func_probe_result="
    %fmt_nl = sim.fmt.literal "\0A"

    // Create a signal with initial value 42
    %sig = llhd.sig %c42_i8 : i8

    llhd.process {
      %delay = llhd.int_to_time %c1_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      // Convert signal ref to LLVM pointer for passing to LLVM function
      %sig_ptr = builtin.unrealized_conversion_cast %sig : !llhd.ref<i8> to !llvm.ptr

      // Call the LLVM function with the signal reference
      %result = llvm.call @probe_signal(%sig_ptr) : (!llvm.ptr) -> i8

      // Print the result (should be 42)
      %fmt_val = sim.fmt.dec %result : i8
      %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
