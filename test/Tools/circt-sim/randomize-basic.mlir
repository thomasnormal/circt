// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that __moore_randomize_basic fills the object with random bytes.
//
// __moore_randomize_basic fills the entire object with random bytes.
// The MooreToCore lowering saves non-rand fields BEFORE this call and restores
// them AFTERWARDS, so it is safe to overwrite the entire object.  Unconstrained
// rand fields that have no explicit __moore_randomize_with_range call get their
// randomness from this fill.  Constrained rand fields are later overridden by
// subsequent _with_range / _with_dist calls.
//
// We allocate a 16-byte block, write known values, call __moore_randomize_basic,
// then verify the values were randomized (overwritten with random data).

// CHECK: randomize_returned = 1
// CHECK: values_randomized = 1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Allocate 16 bytes
      %c16 = arith.constant 16 : i64
      %ptr = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr

      // Write known sentinel values to all 4 words (0x12345678)
      %sentinel = arith.constant 305419896 : i32
      %f0 = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %sentinel, %f0 : i32, !llvm.ptr
      %f1 = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %sentinel, %f1 : i32, !llvm.ptr
      %f2 = llvm.getelementptr %ptr[0, 2]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %sentinel, %f2 : i32, !llvm.ptr
      %f3 = llvm.getelementptr %ptr[0, 3]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %sentinel, %f3 : i32, !llvm.ptr

      // Call __moore_randomize_basic (should be a no-op preserving memory)
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c16) : (!llvm.ptr, i64) -> i32

      // Print return code
      %lit_rc = sim.fmt.literal "randomize_returned = "
      %d_rc = sim.fmt.dec %rc signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt_rc = sim.fmt.concat (%lit_rc, %d_rc, %nl)
      sim.proc.print %fmt_rc

      // Load all 4 words and check at least one was changed (randomized)
      %v0 = llvm.load %f0 : !llvm.ptr -> i32
      %v1 = llvm.load %f1 : !llvm.ptr -> i32
      %v2 = llvm.load %f2 : !llvm.ptr -> i32
      %v3 = llvm.load %f3 : !llvm.ptr -> i32
      %ne0 = comb.icmp ne %v0, %sentinel : i32
      %ne1 = comb.icmp ne %v1, %sentinel : i32
      %ne2 = comb.icmp ne %v2, %sentinel : i32
      %ne3 = comb.icmp ne %v3, %sentinel : i32
      %any_ne01 = comb.or %ne0, %ne1 : i1
      %any_ne23 = comb.or %ne2, %ne3 : i1
      %any_ne = comb.or %any_ne01, %any_ne23 : i1

      // Convert i1 to i32 for printing
      %zero32 = arith.constant 0 : i32
      %one32 = arith.constant 1 : i32
      %result = arith.select %any_ne, %one32, %zero32 : i32

      %lit_pres = sim.fmt.literal "values_randomized = "
      %d_pres = sim.fmt.dec %result signed : i32
      %fmt_pres = sim.fmt.concat (%lit_pres, %d_pres, %nl)
      sim.proc.print %fmt_pres

      llvm.call @free(%ptr) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
