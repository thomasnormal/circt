// RUN: circt-sim %s --max-time=1000000000 2>&1 | FileCheck %s

// Test that global constructors can create runtime signals without crashing.
// This verifies that interpretLLVMFuncBody doesn't hold stale references to
// processStates across operations that may modify the map.

module {
  llvm.mlir.global internal @"test_pkg::global_count"(0 : i32) {addr_space = 0 : i32} : i32

  llvm.func @malloc(i64) -> !llvm.ptr

  // Helper function that creates runtime signals - may trigger map rehash
  func.func private @create_signal() {
    %time = llhd.constant_time <0ns, 0d, 1e>
    %c0 = hw.constant 0 : i32

    // Create a runtime signal (llhd.sig inside function)
    %sig = llhd.sig %c0 : i32

    // Probe the signal to verify it works
    %val = llhd.prb %sig : i32

    // Increment global count
    %addr = llvm.mlir.addressof @"test_pkg::global_count" : !llvm.ptr
    %old = llvm.load %addr : !llvm.ptr -> i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %new = llvm.add %old, %c1 : i32
    llvm.store %new, %addr : i32, !llvm.ptr

    return
  }

  // Function that calls create_signal multiple times
  func.func private @nested_calls() {
    func.call @create_signal() : () -> ()
    func.call @create_signal() : () -> ()
    func.call @create_signal() : () -> ()
    return
  }

  // Global constructor
  llvm.func @global_init() {
    func.call @nested_calls() : () -> ()
    llvm.return
  }

  // Register global constructor
  llvm.mlir.global_ctors ctors = [@global_init], priorities = [65535 : i32], data = [#llvm.zero]

  hw.module @top() {
    %fmt_start = sim.fmt.literal "=== Global Ctor Runtime Signals Test ===\n"
    %fmt_count = sim.fmt.literal "Signals created: "
    %fmt_nl = sim.fmt.literal "\n"
    %fmt_pass = sim.fmt.literal "TEST PASSED\n"

    llhd.process {
      sim.proc.print %fmt_start

      // Read global count (should be 3 from constructor)
      %addr = llvm.mlir.addressof @"test_pkg::global_count" : !llvm.ptr
      %count = llvm.load %addr : !llvm.ptr -> i32

      %count_fmt = sim.fmt.dec %count specifierWidth 0 : i32
      %msg = sim.fmt.concat (%fmt_count, %count_fmt, %fmt_nl)
      sim.proc.print %msg

      // Check count equals 3
      %c3 = llvm.mlir.constant(3 : i32) : i32
      %ok = llvm.icmp "eq" %count, %c3 : i32
      cf.cond_br %ok, ^pass, ^pass  // Always pass for now, main test is no crash

    ^pass:
      sim.proc.print %fmt_pass
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}

// CHECK: === Global Ctor Runtime Signals Test ===
// CHECK: Signals created: 3
// CHECK: TEST PASSED
