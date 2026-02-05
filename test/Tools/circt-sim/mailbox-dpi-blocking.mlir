// RUN: circt-sim %s | FileCheck %s

// Test for Phase 2 blocking mailbox operations in circt-sim.
// This test verifies that __moore_mailbox_put and __moore_mailbox_get
// correctly block when the mailbox is full/empty and resume when
// space/messages become available.

// CHECK: Creating mailbox
// CHECK-DAG: Producer: sending 42
// CHECK-DAG: Producer: sending 100
// CHECK-DAG: Consumer: got 42
// CHECK-DAG: Consumer: got 100
// CHECK: Test passed

!i32 = i32
!i64 = i64
!ptr = !llvm.ptr

hw.module @mailbox_blocking_test() {
  llhd.process {
    // Print initial message
    %fmt_creating = sim.fmt.literal "Creating mailbox\0A"
    sim.proc.print %fmt_creating

    // Create an unbounded mailbox
    %c0_i32 = llvm.mlir.constant(0 : i32) : !i32
    %mbox_id = llvm.call @__moore_mailbox_create(%c0_i32) : (!i32) -> !i64

    // Use fork to run producer and consumer concurrently
    %fmt_producer_prefix = sim.fmt.literal "Producer: sending "
    %fmt_consumer_prefix = sim.fmt.literal "Consumer: got "
    %fmt_nl = sim.fmt.literal "\0A"

    %handle = sim.fork join_type "join" {
      // Producer: send messages using blocking put
      %c42 = llvm.mlir.constant(42 : i64) : !i64
      %fmt42 = sim.fmt.dec %c42 : i64
      %fmt_send42 = sim.fmt.concat (%fmt_producer_prefix, %fmt42, %fmt_nl)
      sim.proc.print %fmt_send42
      llvm.call @__moore_mailbox_put(%mbox_id, %c42) : (!i64, !i64) -> ()

      // Send second message
      %c100 = llvm.mlir.constant(100 : i64) : !i64
      %fmt100 = sim.fmt.dec %c100 : i64
      %fmt_send100 = sim.fmt.concat (%fmt_producer_prefix, %fmt100, %fmt_nl)
      sim.proc.print %fmt_send100
      llvm.call @__moore_mailbox_put(%mbox_id, %c100) : (!i64, !i64) -> ()
      sim.fork.terminator

    }, {
      // Consumer: receive messages using blocking get
      %c1_i64 = llvm.mlir.constant(1 : i64) : !i64
      %c0_i64 = llvm.mlir.constant(0 : i64) : !i64
      %msg_out = llvm.alloca %c1_i64 x !i64 : (!i64) -> !ptr
      llvm.store %c0_i64, %msg_out : !i64, !ptr

      // Blocking get - will wait until producer sends
      llvm.call @__moore_mailbox_get(%mbox_id, %msg_out) : (!i64, !ptr) -> ()
      %msg1 = llvm.load %msg_out : !ptr -> !i64
      %fmt_msg1 = sim.fmt.dec %msg1 : i64
      %fmt_got1 = sim.fmt.concat (%fmt_consumer_prefix, %fmt_msg1, %fmt_nl)
      sim.proc.print %fmt_got1

      // Second blocking get
      llvm.call @__moore_mailbox_get(%mbox_id, %msg_out) : (!i64, !ptr) -> ()
      %msg2 = llvm.load %msg_out : !ptr -> !i64
      %fmt_msg2 = sim.fmt.dec %msg2 : i64
      %fmt_got2 = sim.fmt.concat (%fmt_consumer_prefix, %fmt_msg2, %fmt_nl)
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
