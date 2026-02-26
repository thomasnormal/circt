// RUN: circt-sim %s --top=top --sim-stats 2>&1 | FileCheck %s
//
// Regression: mixed wait_condition modes must not leave stale timed polls.
// A phase wait_for_state() arm uses memory + timed polling fallback (1 ps).
// If it wakes early and later switches to queue-backed event-only waiting,
// stale timed callbacks must be canceled so final time stays at real wake time.
//
// CHECK: done
// CHECK: [circt-sim] Simulation completed at time 7 fs

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_delay(i64)
  llvm.func @__moore_queue_size(i64) -> i32
  llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)

  llvm.mlir.global internal @phase_state(0 : i32) : i32
  llvm.mlir.global internal @queue_header(0 : i128) : i128

  llvm.func @"uvm_pkg::uvm_phase::wait_for_state"(%phase: i64, %target: i32) {
    %stateAddr = llvm.mlir.addressof @phase_state : !llvm.ptr
    %state = llvm.load %stateAddr : !llvm.ptr -> i32
    %cond = comb.icmp eq %state, %target : i32
    %cond_i32 = llvm.zext %cond : i1 to i32
    llvm.call @__moore_wait_condition(%cond_i32) : (i32) -> ()
    llvm.return
  }

  hw.module @top() {
    %fmt_done = sim.fmt.literal "done\0A"

    // Step 1: wait_for_state uses memory + timed polling fallback.
    // Step 2: wait for queue-not-empty (event wakeup only).
    llhd.process {
      %phase = llvm.mlir.constant(42 : i64) : i64
      %target = llvm.mlir.constant(1 : i32) : i32
      llvm.call @"uvm_pkg::uvm_phase::wait_for_state"(%phase, %target) :
          (i64, i32) -> ()

      %queue_ptr = llvm.mlir.addressof @queue_header : !llvm.ptr
      %queue_i64 = llvm.ptrtoint %queue_ptr : !llvm.ptr to i64
      %size = llvm.call @__moore_queue_size(%queue_i64) : (i64) -> i32
      %zero = hw.constant 0 : i32
      %non_empty = comb.icmp ne %size, %zero : i32
      %non_empty_i32 = llvm.zext %non_empty : i1 to i32
      llvm.call @__moore_wait_condition(%non_empty_i32) : (i32) -> ()

      sim.proc.print %fmt_done
      llhd.halt
    }

    // Wake the wait_for_state() wait at 5 fs.
    llhd.process {
      %delay = llvm.mlir.constant(5 : i64) : i64
      %stateAddr = llvm.mlir.addressof @phase_state : !llvm.ptr
      %one = llvm.mlir.constant(1 : i32) : i32
      llvm.call @__moore_delay(%delay) : (i64) -> ()
      llvm.store %one, %stateAddr : i32, !llvm.ptr
      llhd.halt
    }

    // Wake queue wait at 7 fs.
    llhd.process {
      %delay = llvm.mlir.constant(7 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %elem_size = llvm.mlir.constant(8 : i64) : i64
      %value = llvm.mlir.constant(99 : i64) : i64
      llvm.call @__moore_delay(%delay) : (i64) -> ()
      %queue_ptr = llvm.mlir.addressof @queue_header : !llvm.ptr
      %elem_ptr = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
      llvm.store %value, %elem_ptr : i64, !llvm.ptr
      llvm.call @__moore_queue_push_back(%queue_ptr, %elem_ptr, %elem_size) :
          (!llvm.ptr, !llvm.ptr, i64) -> ()
      llhd.halt
    }

    hw.output
  }
}
