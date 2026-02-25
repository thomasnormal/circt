// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression for sub-byte/non-byte-aligned llvm.mlir.global initializers.
// initializeGlobals must not request invalid APInt::extractBits widths when
// packing globals such as i1/i9 into memory.

// CHECK: [circt-sim] Starting simulation
// CHECK: subbyte globals init
// CHECK: g_i1=1
// CHECK: g_i9=257
// CHECK: [circt-sim] Simulation completed

module {
  llvm.mlir.global internal @"test_pkg::g_i1"(true) : i1
  llvm.mlir.global internal @"test_pkg::g_i9"(257 : i9) : i9

  hw.module @top() {
    %fmt_start = sim.fmt.literal "subbyte globals init\0A"
    %fmt_i1 = sim.fmt.literal "g_i1="
    %fmt_i9 = sim.fmt.literal "g_i9="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      sim.proc.print %fmt_start

      %addr_i1 = llvm.mlir.addressof @"test_pkg::g_i1" : !llvm.ptr
      %val_i1 = llvm.load %addr_i1 : !llvm.ptr -> i1
      %val_i1_i32 = llvm.zext %val_i1 : i1 to i32
      %fmt_i1_val = sim.fmt.dec %val_i1_i32 : i32
      %msg_i1 = sim.fmt.concat (%fmt_i1, %fmt_i1_val, %fmt_nl)
      sim.proc.print %msg_i1

      %addr_i9 = llvm.mlir.addressof @"test_pkg::g_i9" : !llvm.ptr
      %val_i9 = llvm.load %addr_i9 : !llvm.ptr -> i9
      %val_i9_i32 = llvm.zext %val_i9 : i9 to i32
      %fmt_i9_val = sim.fmt.dec %val_i9_i32 : i32
      %msg_i9 = sim.fmt.concat (%fmt_i9, %fmt_i9_val, %fmt_nl)
      sim.proc.print %msg_i9

      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
