// RUN: circt-verilog %s --no-uvm-auto-include --ir-moore -o - 2>&1 | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s --no-uvm-auto-include -o - 2>&1 | FileCheck %s

// Test case for array locator with virtual method call in predicate
// This replicates the pattern from uvm_sequencer_base.svh line 1176:
//   q=lock_list.find_first_index(item) with (item.get_inst_id() == seqid);

class uvm_sequence_base;
  virtual function int get_inst_id();
    return 42;
  endfunction
endclass

class uvm_sequencer_base;
  uvm_sequence_base lock_list[$];

  // Function that uses array locator with virtual method call in predicate
  function void m_unlock_req(int seqid);
    int q[$];
    // This pattern matches UVM's usage: virtual method call on array element
    q = lock_list.find_first_index(item) with (item.get_inst_id() == seqid);
  endfunction
endclass

module top;
  uvm_sequencer_base seq;
  initial begin
    seq = new;
    seq.m_unlock_req(123);
  end
endmodule

// MOORE: func.func private @"uvm_sequencer_base::m_unlock_req"
// MOORE: moore.vtable.load_method
// MOORE: func.call_indirect

// CHECK: llvm.getelementptr
// CHECK: llvm.load
// CHECK: llvm.call

// CHECK-NOT: error:
// CHECK-NOT: failed to legalize
