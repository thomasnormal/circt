// RUN: circt-sim %s | FileCheck %s

// Test function result caching for hot UVM phase traversal functions.
// Functions matching uvm_phase:: patterns (get_schedule, find, etc.) have
// their results cached to avoid exponential-time DFS traversals.

// CHECK: sum=30

// A UVM-phase-like function that sums integers. Its name matches the caching
// pattern "uvm_phase::get_schedule" so results will be cached.
func.func private @"test_pkg::uvm_phase::get_schedule"(%arg0: i32) -> i32 {
  %c10 = hw.constant 10 : i32
  %result = comb.add %arg0, %c10 : i32
  return %result : i32
}

hw.module @test() {
  llhd.process {
    // Call the "uvm_phase" function 3 times with the same argument.
    // With caching, only the first call executes; the other two return cached.
    %c0 = hw.constant 0 : i32
    %r1 = func.call @"test_pkg::uvm_phase::get_schedule"(%c0) : (i32) -> i32
    %r2 = func.call @"test_pkg::uvm_phase::get_schedule"(%c0) : (i32) -> i32
    %r3 = func.call @"test_pkg::uvm_phase::get_schedule"(%c0) : (i32) -> i32

    // All three should return 10, so sum = 30
    %sum12 = comb.add %r1, %r2 : i32
    %sum = comb.add %sum12, %r3 : i32

    %fmt_prefix = sim.fmt.literal "sum="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %sum : i32
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
