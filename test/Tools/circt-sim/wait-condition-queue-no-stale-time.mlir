// RUN: circt-sim %s --top=top --sim-stats 2>&1 | FileCheck %s
//
// Regression: queue-backed wait(condition) previously scheduled a sparse
// timed fallback poll (10 us). If the queue wakeup happened earlier, the stale
// fallback event could still advance final simulation time to 10 us.
//
// CHECK: done
// CHECK: [circt-sim] Simulation completed at time 5 fs

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_queue_size(i64) -> i32
  llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)

  llvm.mlir.global internal @queue_header(0 : i128) : i128

  hw.module @top() {
    %delay5 = llhd.constant_time <5fs, 0d, 0e>
    %fmt_done = sim.fmt.literal "done\0A"

    // Wait until queue size becomes non-zero.
    llhd.process {
      %queue_ptr = llvm.mlir.addressof @queue_header : !llvm.ptr
      %queue_i64 = llvm.ptrtoint %queue_ptr : !llvm.ptr to i64
      %size = llvm.call @__moore_queue_size(%queue_i64) : (i64) -> i32
      %c0 = hw.constant 0 : i32
      %cond = comb.icmp ne %size, %c0 : i32
      %cond_i32 = llvm.zext %cond : i1 to i32
      llvm.call @__moore_wait_condition(%cond_i32) : (i32) -> ()
      sim.proc.print %fmt_done
      llhd.halt
    }

    // Push one element after 5 fs to wake the waiter.
    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %elem_size = llvm.mlir.constant(8 : i64) : i64
      %value = llvm.mlir.constant(42 : i64) : i64
      llhd.wait delay %delay5, ^bb1
    ^bb1:
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
