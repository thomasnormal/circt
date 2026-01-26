// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// TLM Analysis Port Queue Foreach Test - Iteration 121 Track D
//===----------------------------------------------------------------------===//
//
// This test verifies that foreach loops over queues (as used in TLM analysis
// ports) use size-based iteration rather than associative array iteration.
//
// The UVM uvm_analysis_port::write() method uses:
//   foreach (m_subscribers[i]) m_subscribers[i].write(t);
//
// This must generate code using array.size and index iteration, NOT
// assoc.first/assoc.next which is only for associative arrays.
//
//===----------------------------------------------------------------------===//

// Interface for a simple subscriber pattern
interface class subscriber_if;
  pure virtual function void write(int data);
endclass

// Simple subscriber implementation
class simple_subscriber implements subscriber_if;
  int received_data;

  virtual function void write(int data);
    received_data = data;
  endfunction
endclass

// Analysis port pattern (similar to uvm_analysis_port)
class analysis_port;
  // Queue of subscribers - key pattern from TLM
  subscriber_if subscribers[$];

  function void connect(subscriber_if sub);
    subscribers.push_back(sub);
  endfunction

  // The write method broadcasts to all subscribers via foreach
  // This foreach MUST use size-based iteration, not assoc array iteration
  function void write(int data);
    foreach (subscribers[i])
      subscribers[i].write(data);
  endfunction
endclass

// CHECK: func.func private @"analysis_port::write"
// CHECK-SAME: %arg0: !moore.class<@analysis_port>
// CHECK-SAME: %arg1: !moore.i32

// Queue foreach should produce size-based iteration:
// CHECK: %[[REF:.*]] = moore.class.property_ref %arg0[@subscribers]
// CHECK: cf.br ^bb1(%{{.*}} : !moore.i32)
// CHECK: ^bb1(%[[IDX:.*]]: !moore.i32)
// CHECK: %[[Q:.*]] = moore.read %[[REF]]
// CHECK: %[[SIZE:.*]] = moore.array.size %[[Q]]
// CHECK: %[[CMP:.*]] = moore.slt %[[IDX]], %[[SIZE]]
// CHECK: cf.cond_br

// NOT associative array iteration:
// CHECK-NOT: moore.assoc.first{{.*}}subscribers
// CHECK-NOT: moore.assoc.next{{.*}}subscribers

module tlm_analysis_port_test;
  initial begin
    analysis_port ap = new();
    simple_subscriber sub1 = new();
    simple_subscriber sub2 = new();

    ap.connect(sub1);
    ap.connect(sub2);
    ap.write(42);

    $display("sub1 received: %d", sub1.received_data);
    $display("sub2 received: %d", sub2.received_data);
  end
endmodule

// CHECK: moore.module @tlm_analysis_port_test
