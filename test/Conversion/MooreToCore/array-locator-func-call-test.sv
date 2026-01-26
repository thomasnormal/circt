// RUN: circt-verilog %s --ir-moore -o - 2>&1 | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s -o - 2>&1 | FileCheck %s
// XFAIL: *
// Array locator with function call in predicate needs top module to emit func.

// Test case for array locator with external function call in predicate
// This replicates the pattern from uvm_sequencer_base.svh line 607:
//   lock_req_indices = arb_sequence_q.find_first_index(item) with
//     (item.request==SEQ_TYPE_LOCK && is_blocked(item.sequence_ptr) == 0);

class uvm_sequence_base;
  int id;
endclass

class uvm_sequence_request;
  int request;
  uvm_sequence_base sequence_ptr;
endclass

class uvm_sequencer_base;
  uvm_sequence_request arb_sequence_q[$];

  // Helper function called within the predicate
  function bit is_blocked(uvm_sequence_base seq_ptr);
    return seq_ptr.id == 0;
  endfunction

  // Function that uses array locator with function call in predicate
  function void m_update_lists();
    int lock_req_indices[$];
    // This pattern matches UVM's usage: field comparison AND function call
    lock_req_indices = arb_sequence_q.find_first_index(item) with
      (item.request == 1 && is_blocked(item.sequence_ptr) == 0);
  endfunction
endclass

// MOORE: func.func private @"uvm_sequencer_base::m_update_lists"
// MOORE: moore.array.locator
// MOORE: func.call @"uvm_sequencer_base::is_blocked"

// CHECK-NOT: error:
// CHECK-NOT: failed to legalize
