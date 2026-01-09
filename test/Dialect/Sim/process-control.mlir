// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Fork/Join Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fork_join_basic
func.func @fork_join_basic() {
  // CHECK: %[[HANDLE:.*]] = sim.fork {
  // CHECK: }, {
  // CHECK: }
  %handle = sim.fork {
    sim.fork.terminator
  }, {
    sim.fork.terminator
  }
  // CHECK: sim.join %[[HANDLE]]
  sim.join %handle
  return
}

// CHECK-LABEL: func.func @fork_join_any
func.func @fork_join_any() {
  // CHECK: %[[HANDLE:.*]] = sim.fork join_type "join_any"
  %handle = sim.fork join_type "join_any" {
    sim.fork.terminator
  }, {
    sim.fork.terminator
  }
  // CHECK: sim.join_any %[[HANDLE]]
  sim.join_any %handle
  return
}

// CHECK-LABEL: func.func @fork_join_none
func.func @fork_join_none() {
  // CHECK: %[[HANDLE:.*]] = sim.fork join_type "join_none"
  %handle = sim.fork join_type "join_none" {
    sim.fork.terminator
  }
  // CHECK: sim.join_none %[[HANDLE]]
  sim.join_none %handle
  return
}

// CHECK-LABEL: func.func @wait_fork_basic
func.func @wait_fork_basic() {
  %h1 = sim.fork join_type "join_none" {
    sim.fork.terminator
  }
  %h2 = sim.fork join_type "join_none" {
    sim.fork.terminator
  }
  // CHECK: sim.wait_fork
  sim.wait_fork
  return
}

//===----------------------------------------------------------------------===//
// Wait Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @wait_condition
func.func @wait_condition(%cond: i1) {
  // CHECK: sim.wait %arg0
  sim.wait %cond
  return
}

// CHECK-LABEL: func.func @wait_timeout
func.func @wait_timeout(%cond: i1) {
  // CHECK: %[[TIMED_OUT:.*]] = sim.wait %arg0 timeout 1000000000 : i1
  %timed_out = sim.wait %cond timeout 1000000000 : i1
  return
}

// CHECK-LABEL: func.func @delay_op
func.func @delay_op(%time: i64) {
  // CHECK: sim.delay %arg0 : i64
  sim.delay %time : i64
  return
}

//===----------------------------------------------------------------------===//
// Disable Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @named_block_basic
func.func @named_block_basic() {
  // CHECK: sim.named_block "my_block"
  sim.named_block "my_block" {
    sim.named_block.terminator
  }
  return
}

// CHECK-LABEL: func.func @disable_basic
func.func @disable_basic() {
  sim.named_block "target_block" {
    // CHECK: sim.disable "target_block"
    sim.disable "target_block"
    sim.named_block.terminator
  }
  return
}

// CHECK-LABEL: func.func @disable_fork_op
func.func @disable_fork_op() {
  %h = sim.fork join_type "join_none" {
    sim.fork.terminator
  }
  // CHECK: sim.disable_fork
  sim.disable_fork
  return
}

//===----------------------------------------------------------------------===//
// Semaphore Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @semaphore_create
func.func @semaphore_create() {
  %c1 = arith.constant 1 : i64
  // CHECK: %[[SEM:.*]] = sim.semaphore.create %c1_i64 : i64
  %sem = sim.semaphore.create %c1 : i64
  return
}

// CHECK-LABEL: func.func @semaphore_operations
func.func @semaphore_operations(%sem: i64, %count: i32) {
  // CHECK: sim.semaphore.get %arg0
  sim.semaphore.get %sem

  // CHECK: sim.semaphore.get %arg0, %arg1 : i32
  sim.semaphore.get %sem, %count : i32

  // CHECK: %[[SUCCESS:.*]] = sim.semaphore.try_get %arg0 -> i1
  %success = sim.semaphore.try_get %sem -> i1

  // CHECK: %[[SUCCESS2:.*]] = sim.semaphore.try_get %arg0, %arg1 : i32 -> i1
  %success2 = sim.semaphore.try_get %sem, %count : i32 -> i1

  // CHECK: sim.semaphore.put %arg0
  sim.semaphore.put %sem

  // CHECK: sim.semaphore.put %arg0, %arg1 : i32
  sim.semaphore.put %sem, %count : i32

  return
}

//===----------------------------------------------------------------------===//
// Mailbox Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mailbox_create
func.func @mailbox_create() {
  // CHECK: %[[MBOX:.*]] = sim.mailbox.create : i64
  %mbox = sim.mailbox.create : i64

  // CHECK: %[[BOUNDED:.*]] = sim.mailbox.create bound 16 : i64
  %bounded = sim.mailbox.create bound 16 : i64

  return
}

// CHECK-LABEL: func.func @mailbox_operations
func.func @mailbox_operations(%mbox: i64, %data: i32) {
  // CHECK: sim.mailbox.put %arg0, %arg1 : i32
  sim.mailbox.put %mbox, %data : i32

  // CHECK: %[[SUCCESS:.*]] = sim.mailbox.try_put %arg0, %arg1 : i32 -> i1
  %success = sim.mailbox.try_put %mbox, %data : i32 -> i1

  // CHECK: %[[MSG:.*]] = sim.mailbox.get %arg0 : i32
  %msg = sim.mailbox.get %mbox : i32

  // CHECK: %[[SUCCESS2:.*]], %[[MSG2:.*]] = sim.mailbox.try_get %arg0 : i1, i32
  %success2, %msg2 = sim.mailbox.try_get %mbox : i1, i32

  // CHECK: %[[PEEK:.*]] = sim.mailbox.peek %arg0 : i32
  %peek = sim.mailbox.peek %mbox : i32

  // CHECK: %[[SUCCESS3:.*]], %[[PEEK2:.*]] = sim.mailbox.try_peek %arg0 : i1, i32
  %success3, %peek2 = sim.mailbox.try_peek %mbox : i1, i32

  // CHECK: %[[COUNT:.*]] = sim.mailbox.num %arg0 : i32
  %count = sim.mailbox.num %mbox : i32

  return
}

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @event_operations
func.func @event_operations() {
  // CHECK: %[[EVT:.*]] = sim.event.create : i64
  %evt = sim.event.create : i64

  // CHECK: sim.event.trigger %[[EVT]]
  sim.event.trigger %evt

  // CHECK: sim.event.trigger %[[EVT]] nonblocking
  sim.event.trigger %evt nonblocking

  // CHECK: sim.event.wait %[[EVT]]
  sim.event.wait %evt

  return
}
