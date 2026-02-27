// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top fork_wait_event_parent_memory_tb --max-time=200000000 2>&1 | FileCheck %s

// Regression: memory-backed wait_event in a fork child must wake when the
// watched storage lives in the parent process frame.
//
// Pattern:
// 1. Parent initial block allocates a local variable.
// 2. Fork child A waits on @(posedge ref_var) via task ref argument.
// 3. Fork child B updates the same parent-local variable.
//
// If parent-frame memory is not visible to memory-event polling, child A never
// wakes and "DONE" is not printed.

// CHECK: CHILD_WOKE
// CHECK: DONE
// CHECK-NOT: maxTime reached
module fork_wait_event_parent_memory_tb;
  task automatic wait_posedge_ref(ref logic sig);
    @(posedge sig);
    $display("CHILD_WOKE");
  endtask

  initial begin
    logic local_sig;
    local_sig = 1'b0;

    fork
      wait_posedge_ref(local_sig);
      begin
        #5;
        local_sig = 1'b1;
      end
    join

    $display("DONE");
  end
endmodule

