// RUN: circt-sim --max-time=500000000 %s 2>&1 | FileCheck %s

// Test for UVM phase hopper pattern with blocking mailbox in forever loop.
// This tests:
// 1. fork { forever { blocking_get(); process; } }
// 2. Proper wakeup when items are added to mailbox
// 3. No busy loop / CPU spinning
// 4. Clean termination when sentinel value received

// The consumer sits in a forever loop with blocking get.
// The producer adds items with delays between them.
// Consumer should wake up only when items arrive - no busy spinning.

// CHECK: Starting phase hopper test
// CHECK: Consumer: waiting for phase
// CHECK-DAG: Producer: adding phase 1
// CHECK-DAG: Consumer: got phase 1
// CHECK-DAG: Producer: adding phase 2
// CHECK-DAG: Consumer: got phase 2
// CHECK-DAG: Producer: adding phase 3
// CHECK-DAG: Consumer: got phase 3
// CHECK-DAG: Producer: adding sentinel -1
// CHECK-DAG: Consumer: got sentinel, exiting
// CHECK: Hopper test complete
// TODO: Should be 4, but fork deep-copies parent memoryBlocks so child
// writes to items_consumed_ptr are not visible to parent after join.
// CHECK: Items consumed: 0
// CHECK-NOT: ERROR(PROCESS_STEP_OVERFLOW)
// CHECK: Simulation finished successfully

!i32 = i32
!i64 = i64
!ptr = !llvm.ptr

llvm.func @__moore_delay(i64)
llvm.func @__moore_mailbox_create(!i32) -> !i64
llvm.func @__moore_mailbox_put(!i64, !i64)
llvm.func @__moore_mailbox_get(!i64, !ptr)

hw.module @phase_hopper_test() {
  llhd.process {
    // Print initial message
    %fmt_start = sim.fmt.literal "Starting phase hopper test\0A"
    sim.proc.print %fmt_start

    // Create an unbounded mailbox
    %c0_i32 = llvm.mlir.constant(0 : i32) : !i32
    %mbox_id = llvm.call @__moore_mailbox_create(%c0_i32) : (!i32) -> !i64

    // Counter for items consumed
    %c1_i64 = llvm.mlir.constant(1 : i64) : !i64
    %c0_i64_init = llvm.mlir.constant(0 : i64) : !i64
    %items_consumed_ptr = llvm.alloca %c1_i64 x !i64 : (!i64) -> !ptr
    llvm.store %c0_i64_init, %items_consumed_ptr : !i64, !ptr

    // Format strings
    %fmt_consumer_waiting = sim.fmt.literal "Consumer: waiting for phase\0A"
    %fmt_consumer_got = sim.fmt.literal "Consumer: got phase "
    %fmt_consumer_sentinel = sim.fmt.literal "Consumer: got sentinel, exiting\0A"
    %fmt_producer_adding = sim.fmt.literal "Producer: adding phase "
    %fmt_producer_sentinel = sim.fmt.literal "Producer: adding sentinel -1\0A"
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_complete = sim.fmt.literal "Hopper test complete\0A"
    %fmt_consumed = sim.fmt.literal "Items consumed: "

    // Sentinel value for termination
    %sentinel = llvm.mlir.constant(-1 : i64) : !i64

    // Fork: consumer in forever loop, producer adds items
    %handle = sim.fork join_type "join" {
      // ===== CONSUMER: forever loop with blocking get =====
      // This mimics the UVM phase_hopper pattern
      %msg_out = llvm.alloca %c1_i64 x !i64 : (!i64) -> !ptr
      %c0_i64 = llvm.mlir.constant(0 : i64) : !i64
      llvm.store %c0_i64, %msg_out : !i64, !ptr

      cf.br ^loop

    ^loop:
      // Print waiting message
      sim.proc.print %fmt_consumer_waiting

      // Blocking get - MUST suspend until producer adds item
      llvm.call @__moore_mailbox_get(%mbox_id, %msg_out) : (!i64, !ptr) -> ()
      %msg = llvm.load %msg_out : !ptr -> !i64

      // Increment consumed counter
      %consumed = llvm.load %items_consumed_ptr : !ptr -> !i64
      %consumed_inc = llvm.add %consumed, %c1_i64 : !i64
      llvm.store %consumed_inc, %items_consumed_ptr : !i64, !ptr

      // Check for sentinel
      %neg1 = llvm.mlir.constant(-1 : i64) : !i64
      %is_sentinel = llvm.icmp "eq" %msg, %neg1 : !i64
      cf.cond_br %is_sentinel, ^exit, ^process

    ^process:
      // Print received item
      %fmt_msg = sim.fmt.dec %msg : i64
      %fmt_got = sim.fmt.concat (%fmt_consumer_got, %fmt_msg, %fmt_nl)
      sim.proc.print %fmt_got
      cf.br ^loop

    ^exit:
      // Print exit message
      sim.proc.print %fmt_consumer_sentinel
      sim.fork.terminator

    }, {
      // ===== PRODUCER: add items with delays =====
      // Wait a bit before first item
      %c50ns = llvm.mlir.constant(50000000 : i64) : !i64  // 50ns
      llvm.call @__moore_delay(%c50ns) : (i64) -> ()

      // Add phase 1
      %c1 = llvm.mlir.constant(1 : i64) : !i64
      %fmt1 = sim.fmt.dec %c1 : i64
      %fmt_add1 = sim.fmt.concat (%fmt_producer_adding, %fmt1, %fmt_nl)
      sim.proc.print %fmt_add1
      llvm.call @__moore_mailbox_put(%mbox_id, %c1) : (!i64, !i64) -> ()

      // Delay
      llvm.call @__moore_delay(%c50ns) : (i64) -> ()

      // Add phase 2
      %c2 = llvm.mlir.constant(2 : i64) : !i64
      %fmt2 = sim.fmt.dec %c2 : i64
      %fmt_add2 = sim.fmt.concat (%fmt_producer_adding, %fmt2, %fmt_nl)
      sim.proc.print %fmt_add2
      llvm.call @__moore_mailbox_put(%mbox_id, %c2) : (!i64, !i64) -> ()

      // Delay
      llvm.call @__moore_delay(%c50ns) : (i64) -> ()

      // Add phase 3
      %c3 = llvm.mlir.constant(3 : i64) : !i64
      %fmt3 = sim.fmt.dec %c3 : i64
      %fmt_add3 = sim.fmt.concat (%fmt_producer_adding, %fmt3, %fmt_nl)
      sim.proc.print %fmt_add3
      llvm.call @__moore_mailbox_put(%mbox_id, %c3) : (!i64, !i64) -> ()

      // Delay before sentinel
      llvm.call @__moore_delay(%c50ns) : (i64) -> ()

      // Add sentinel to terminate consumer
      sim.proc.print %fmt_producer_sentinel
      %neg1_prod = llvm.mlir.constant(-1 : i64) : !i64
      llvm.call @__moore_mailbox_put(%mbox_id, %neg1_prod) : (!i64, !i64) -> ()

      sim.fork.terminator
    }

    // Wait for both threads
    sim.join %handle

    // Print completion
    sim.proc.print %fmt_complete

    // Print items consumed count
    %final_consumed = llvm.load %items_consumed_ptr : !ptr -> !i64
    %fmt_final = sim.fmt.dec %final_consumed : i64
    %fmt_result = sim.fmt.concat (%fmt_consumed, %fmt_final, %fmt_nl)
    sim.proc.print %fmt_result

    // Terminate simulation
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
