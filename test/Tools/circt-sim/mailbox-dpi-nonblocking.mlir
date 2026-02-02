// RUN: circt-sim %s | FileCheck %s

// Test non-blocking mailbox DPI hooks.
// Phase 1: Tests __moore_mailbox_create, __moore_mailbox_tryput,
//          __moore_mailbox_tryget, and __moore_mailbox_num.

// CHECK: Creating mailbox
// CHECK: Mailbox ID > 0: true
// CHECK: Initial num: 0
// CHECK: tryput(42): true
// CHECK: num: 1
// CHECK: tryput(123): true
// CHECK: num: 2
// CHECK: tryget: true, msg=42
// CHECK: num: 1
// CHECK: tryget: true, msg=123
// CHECK: num: 0
// CHECK: tryget empty: false
// CHECK: Bounded test
// CHECK: bounded put1: true
// CHECK: bounded put2 (full): false
// CHECK: Done

!i1 = i1
!i32 = i32
!i64 = i64
!ptr = !llvm.ptr

hw.module @mailbox_nonblocking_test() {
  llhd.process {
    // Print initial message
    %fmt_creating = sim.fmt.literal "Creating mailbox\0A"
    sim.proc.print %fmt_creating

    // Create an unbounded mailbox (bound = 0)
    %c0_i32 = llvm.mlir.constant(0 : i32) : !i32
    %mbox_id = llvm.call @__moore_mailbox_create(%c0_i32) : (!i32) -> !i64

    // Check that mailbox ID is > 0
    %c0_i64 = llvm.mlir.constant(0 : i64) : !i64
    %id_gt_zero = llvm.icmp "sgt" %mbox_id, %c0_i64 : !i64
    %fmt_id_prefix = sim.fmt.literal "Mailbox ID > 0: "
    %fmt_id_val = sim.fmt.bin %id_gt_zero : i1
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_true_lit = sim.fmt.literal "true"
    %fmt_false_lit = sim.fmt.literal "false"
    scf.if %id_gt_zero {
      %fmt_id = sim.fmt.concat (%fmt_id_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_id
    } else {
      %fmt_id = sim.fmt.concat (%fmt_id_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_id
    }

    // Check initial message count (should be 0)
    %num0 = llvm.call @__moore_mailbox_num(%mbox_id) : (!i64) -> !i64
    %fmt_num_prefix = sim.fmt.literal "Initial num: "
    %fmt_num0_val = sim.fmt.dec %num0 : i64
    %fmt_num0 = sim.fmt.concat (%fmt_num_prefix, %fmt_num0_val, %fmt_nl)
    sim.proc.print %fmt_num0

    // Try to put a message (42)
    %c42 = llvm.mlir.constant(42 : i64) : !i64
    %put1_ok = llvm.call @__moore_mailbox_tryput(%mbox_id, %c42) : (!i64, !i64) -> !i1
    %fmt_put1_prefix = sim.fmt.literal "tryput(42): "
    scf.if %put1_ok {
      %fmt_put1 = sim.fmt.concat (%fmt_put1_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_put1
    } else {
      %fmt_put1 = sim.fmt.concat (%fmt_put1_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_put1
    }

    // Check message count (should be 1)
    %num1 = llvm.call @__moore_mailbox_num(%mbox_id) : (!i64) -> !i64
    %fmt_num1_prefix = sim.fmt.literal "num: "
    %fmt_num1_val = sim.fmt.dec %num1 : i64
    %fmt_num1 = sim.fmt.concat (%fmt_num1_prefix, %fmt_num1_val, %fmt_nl)
    sim.proc.print %fmt_num1

    // Put another message (123)
    %c123 = llvm.mlir.constant(123 : i64) : !i64
    %put2_ok = llvm.call @__moore_mailbox_tryput(%mbox_id, %c123) : (!i64, !i64) -> !i1
    %fmt_put2_prefix = sim.fmt.literal "tryput(123): "
    scf.if %put2_ok {
      %fmt_put2 = sim.fmt.concat (%fmt_put2_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_put2
    } else {
      %fmt_put2 = sim.fmt.concat (%fmt_put2_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_put2
    }

    // Check message count (should be 2)
    %num2 = llvm.call @__moore_mailbox_num(%mbox_id) : (!i64) -> !i64
    %fmt_num2_val = sim.fmt.dec %num2 : i64
    %fmt_num2 = sim.fmt.concat (%fmt_num1_prefix, %fmt_num2_val, %fmt_nl)
    sim.proc.print %fmt_num2

    // Allocate space for received message
    %c1_i64 = llvm.mlir.constant(1 : i64) : !i64
    %msg_out = llvm.alloca %c1_i64 x !i64 : (!i64) -> !ptr
    llvm.store %c0_i64, %msg_out : !i64, !ptr

    // Try to get first message (should be 42, FIFO order)
    %get1_ok = llvm.call @__moore_mailbox_tryget(%mbox_id, %msg_out) : (!i64, !ptr) -> !i1
    %msg1 = llvm.load %msg_out : !ptr -> !i64
    %fmt_get1_prefix = sim.fmt.literal "tryget: "
    %fmt_msg_prefix = sim.fmt.literal ", msg="
    %fmt_msg1_val = sim.fmt.dec %msg1 : i64
    scf.if %get1_ok {
      %fmt_get1 = sim.fmt.concat (%fmt_get1_prefix, %fmt_true_lit, %fmt_msg_prefix, %fmt_msg1_val, %fmt_nl)
      sim.proc.print %fmt_get1
    } else {
      %fmt_get1 = sim.fmt.concat (%fmt_get1_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_get1
    }

    // Check message count (should be 1)
    %num3 = llvm.call @__moore_mailbox_num(%mbox_id) : (!i64) -> !i64
    %fmt_num3_val = sim.fmt.dec %num3 : i64
    %fmt_num3 = sim.fmt.concat (%fmt_num1_prefix, %fmt_num3_val, %fmt_nl)
    sim.proc.print %fmt_num3

    // Try to get second message (should be 123)
    %get2_ok = llvm.call @__moore_mailbox_tryget(%mbox_id, %msg_out) : (!i64, !ptr) -> !i1
    %msg2 = llvm.load %msg_out : !ptr -> !i64
    %fmt_msg2_val = sim.fmt.dec %msg2 : i64
    scf.if %get2_ok {
      %fmt_get2 = sim.fmt.concat (%fmt_get1_prefix, %fmt_true_lit, %fmt_msg_prefix, %fmt_msg2_val, %fmt_nl)
      sim.proc.print %fmt_get2
    } else {
      %fmt_get2 = sim.fmt.concat (%fmt_get1_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_get2
    }

    // Check message count (should be 0)
    %num4 = llvm.call @__moore_mailbox_num(%mbox_id) : (!i64) -> !i64
    %fmt_num4_val = sim.fmt.dec %num4 : i64
    %fmt_num4 = sim.fmt.concat (%fmt_num1_prefix, %fmt_num4_val, %fmt_nl)
    sim.proc.print %fmt_num4

    // Try to get from empty mailbox (should fail)
    %get3_ok = llvm.call @__moore_mailbox_tryget(%mbox_id, %msg_out) : (!i64, !ptr) -> !i1
    %fmt_get_empty_prefix = sim.fmt.literal "tryget empty: "
    scf.if %get3_ok {
      %fmt_get3 = sim.fmt.concat (%fmt_get_empty_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_get3
    } else {
      %fmt_get3 = sim.fmt.concat (%fmt_get_empty_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_get3
    }

    // Test bounded mailbox
    %fmt_bounded = sim.fmt.literal "Bounded test\0A"
    sim.proc.print %fmt_bounded
    %c1_i32 = llvm.mlir.constant(1 : i32) : !i32
    %bounded_mbox = llvm.call @__moore_mailbox_create(%c1_i32) : (!i32) -> !i64

    // First put should succeed
    %bput1_ok = llvm.call @__moore_mailbox_tryput(%bounded_mbox, %c42) : (!i64, !i64) -> !i1
    %fmt_bput1_prefix = sim.fmt.literal "bounded put1: "
    scf.if %bput1_ok {
      %fmt_bput1 = sim.fmt.concat (%fmt_bput1_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_bput1
    } else {
      %fmt_bput1 = sim.fmt.concat (%fmt_bput1_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_bput1
    }

    // Second put should fail (bounded to 1)
    %bput2_ok = llvm.call @__moore_mailbox_tryput(%bounded_mbox, %c123) : (!i64, !i64) -> !i1
    %fmt_bput2_prefix = sim.fmt.literal "bounded put2 (full): "
    scf.if %bput2_ok {
      %fmt_bput2 = sim.fmt.concat (%fmt_bput2_prefix, %fmt_true_lit, %fmt_nl)
      sim.proc.print %fmt_bput2
    } else {
      %fmt_bput2 = sim.fmt.concat (%fmt_bput2_prefix, %fmt_false_lit, %fmt_nl)
      sim.proc.print %fmt_bput2
    }

    %fmt_done = sim.fmt.literal "Done\0A"
    sim.proc.print %fmt_done

    // Terminate simulation
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

// External function declarations
llvm.func @__moore_mailbox_create(!i32) -> !i64
llvm.func @__moore_mailbox_tryput(!i64, !i64) -> !i1
llvm.func @__moore_mailbox_tryget(!i64, !ptr) -> !i1
llvm.func @__moore_mailbox_num(!i64) -> !i64
