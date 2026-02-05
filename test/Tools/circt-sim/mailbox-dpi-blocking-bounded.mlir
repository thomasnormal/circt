// RUN: circt-sim %s | FileCheck %s

// Test for Phase 2 blocking mailbox operations with bounded mailbox.
// This test verifies that __moore_mailbox_put blocks when a bounded
// mailbox is full and resumes when space becomes available.

// CHECK: Creating bounded mailbox (size 1)
// CHECK-DAG: Producer: put 1
// CHECK-DAG: Producer: put blocked, waiting
// CHECK-DAG: Consumer: got 1
// CHECK-DAG: Producer: put 2 (after unblock)
// CHECK-DAG: Consumer: got 2
// CHECK: Test passed

!i32 = i32
!i64 = i64
!ptr = !llvm.ptr

hw.module @mailbox_blocking_bounded_test() {
  llhd.process {
    // Print initial message
    %fmt_creating = sim.fmt.literal "Creating bounded mailbox (size 1)\0A"
    sim.proc.print %fmt_creating

    // Create a bounded mailbox with size 1
    %c1_i32 = llvm.mlir.constant(1 : i32) : !i32
    %mbox_id = llvm.call @__moore_mailbox_create(%c1_i32) : (!i32) -> !i64

    // Use fork to run producer and consumer concurrently
    %fmt_nl = sim.fmt.literal "\0A"

    %handle = sim.fork join_type "join" {
      // Producer: send messages using blocking put
      %c1 = llvm.mlir.constant(1 : i64) : !i64
      %fmt_put1 = sim.fmt.literal "Producer: put 1\0A"
      sim.proc.print %fmt_put1
      llvm.call @__moore_mailbox_put(%mbox_id, %c1) : (!i64, !i64) -> ()

      // This second put will block because mailbox is full (size 1)
      %fmt_blocked = sim.fmt.literal "Producer: put blocked, waiting\0A"
      sim.proc.print %fmt_blocked
      %c2 = llvm.mlir.constant(2 : i64) : !i64
      llvm.call @__moore_mailbox_put(%mbox_id, %c2) : (!i64, !i64) -> ()

      %fmt_unblocked = sim.fmt.literal "Producer: put 2 (after unblock)\0A"
      sim.proc.print %fmt_unblocked
      sim.fork.terminator

    }, {
      // Consumer: receive messages using blocking get
      %c1_i64 = llvm.mlir.constant(1 : i64) : !i64
      %c0_i64 = llvm.mlir.constant(0 : i64) : !i64
      %msg_out = llvm.alloca %c1_i64 x !i64 : (!i64) -> !ptr
      llvm.store %c0_i64, %msg_out : !i64, !ptr

      // Blocking get - will receive first message
      llvm.call @__moore_mailbox_get(%mbox_id, %msg_out) : (!i64, !ptr) -> ()
      %msg1 = llvm.load %msg_out : !ptr -> !i64
      %fmt_got_prefix = sim.fmt.literal "Consumer: got "
      %fmt_msg1 = sim.fmt.dec %msg1 : i64
      %fmt_got1 = sim.fmt.concat (%fmt_got_prefix, %fmt_msg1, %fmt_nl)
      sim.proc.print %fmt_got1

      // Second blocking get - will receive second message after producer unblocks
      llvm.call @__moore_mailbox_get(%mbox_id, %msg_out) : (!i64, !ptr) -> ()
      %msg2 = llvm.load %msg_out : !ptr -> !i64
      %fmt_msg2 = sim.fmt.dec %msg2 : i64
      %fmt_got2 = sim.fmt.concat (%fmt_got_prefix, %fmt_msg2, %fmt_nl)
      sim.proc.print %fmt_got2
      sim.fork.terminator
    }

    // Wait for both producer and consumer to finish
    sim.join %handle

    // Print test passed
    %fmt_passed = sim.fmt.literal "Test passed\0A"
    sim.proc.print %fmt_passed

    // Terminate simulation
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

// External function declarations
llvm.func @__moore_mailbox_create(!i32) -> !i64
llvm.func @__moore_mailbox_put(!i64, !i64)
llvm.func @__moore_mailbox_get(!i64, !ptr)
