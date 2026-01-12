// RUN: circt-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Fork/Join Errors
//===----------------------------------------------------------------------===//

func.func @fork_invalid_join_type() {
  // expected-error @below {{invalid join type 'invalid', expected one of: join, join_any, join_none}}
  %handle = sim.fork join_type "invalid" {
    sim.fork.terminator
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// Wait Errors
//===----------------------------------------------------------------------===//

func.func @wait_timeout_without_result(%cond: i1) {
  // expected-error @below {{wait with timeout must have a timedOut result to capture timeout status}}
  sim.wait %cond timeout 1000000000
  return
}

// -----

func.func @wait_result_without_timeout(%cond: i1) -> i1 {
  // expected-error @below {{wait without timeout should not have a timedOut result}}
  %timed_out = sim.wait %cond : i1
  return %timed_out : i1
}
