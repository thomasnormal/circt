// RUN: circt-sim %s | FileCheck %s

// Verify the targeted m_execute_scheduled_forks no-op fast path.
// The function body increments @counter, but the fast path should bypass it.

module {
  llvm.mlir.global internal @counter(0 : i32) : i32

  func.func private @m_execute_scheduled_forks() {
    %counter_ptr = llvm.mlir.addressof @counter : !llvm.ptr
    %value = llvm.load %counter_ptr : !llvm.ptr -> i32
    %one = hw.constant 1 : i32
    %next = comb.add %value, %one : i32
    llvm.store %next, %counter_ptr : i32, !llvm.ptr
    return
  }

  hw.module @main() {
    %fmtPrefix = sim.fmt.literal "counter = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      func.call @m_execute_scheduled_forks() : () -> ()

      %counter_ptr = llvm.mlir.addressof @counter : !llvm.ptr
      %value = llvm.load %counter_ptr : !llvm.ptr -> i32
      %value_dec = sim.fmt.dec %value signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %value_dec, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: counter = 0
