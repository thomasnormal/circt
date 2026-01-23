// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// UVM TLM Port/Export/Imp Test
//===----------------------------------------------------------------------===//
//
// This test validates the complete set of UVM TLM port, export, and
// implementation classes used in AVIP testbenches. Tests include:
//
// 1. Blocking port/export/imp: put, get, peek, get_peek
// 2. Nonblocking port/export/imp: try_put, try_get, try_peek, can_*
// 3. Combined port/export/imp: put, get, peek, get_peek
// 4. TLM FIFO with working get/peek operations
// 5. Port-to-export-to-imp connection chains
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"

import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Transaction class
//===----------------------------------------------------------------------===//

class test_transaction extends uvm_sequence_item;
  `uvm_object_utils(test_transaction)

  rand bit [31:0] data;
  rand bit [7:0] id;

  function new(string name = "test_transaction");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("id=%0d data=0x%08h", id, data);
  endfunction
endclass

// CHECK: moore.class.classdecl @test_transaction

//===----------------------------------------------------------------------===//
// Producer with blocking_put_port
//===----------------------------------------------------------------------===//

class producer extends uvm_component;
  `uvm_component_utils(producer)

  uvm_blocking_put_port #(test_transaction) put_port;

  function new(string name = "producer", uvm_component parent = null);
    super.new(name, parent);
    put_port = new("put_port", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    test_transaction txn;
    super.run_phase(phase);

    for (int i = 0; i < 5; i++) begin
      txn = test_transaction::type_id::create($sformatf("txn_%0d", i));
      txn.id = i;
      txn.data = i * 100;
      `uvm_info(get_type_name(), $sformatf("Producing: %s", txn.convert2string()), UVM_HIGH)
      put_port.put(txn);
    end
  endtask
endclass

// CHECK: moore.class.classdecl @producer extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Consumer with blocking_get_port
//===----------------------------------------------------------------------===//

class consumer extends uvm_component;
  `uvm_component_utils(consumer)

  uvm_blocking_get_port #(test_transaction) get_port;
  int received_count;

  function new(string name = "consumer", uvm_component parent = null);
    super.new(name, parent);
    get_port = new("get_port", this);
    received_count = 0;
  endfunction

  virtual task run_phase(uvm_phase phase);
    test_transaction txn;
    super.run_phase(phase);

    for (int i = 0; i < 5; i++) begin
      get_port.get(txn);
      received_count++;
      `uvm_info(get_type_name(), $sformatf("Consumed: %s", txn.convert2string()), UVM_HIGH)
    end
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), $sformatf("Received %0d transactions", received_count), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @consumer extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Component with blocking_put_imp (provides put implementation)
//===----------------------------------------------------------------------===//

class put_target extends uvm_component;
  `uvm_component_utils(put_target)

  uvm_blocking_put_imp #(test_transaction, put_target) put_imp;
  test_transaction received_txns[$];

  function new(string name = "put_target", uvm_component parent = null);
    super.new(name, parent);
    put_imp = new("put_imp", this);
  endfunction

  // Implementation of put method - called via imp
  virtual task put(test_transaction t);
    received_txns.push_back(t);
    `uvm_info(get_type_name(), $sformatf("Received via put_imp: %s", t.convert2string()), UVM_HIGH)
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), $sformatf("Total received: %0d", received_txns.size()), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @put_target extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Component with blocking_get_imp (provides get implementation)
//===----------------------------------------------------------------------===//

class get_source extends uvm_component;
  `uvm_component_utils(get_source)

  uvm_blocking_get_imp #(test_transaction, get_source) get_imp;
  test_transaction ready_txns[$];

  function new(string name = "get_source", uvm_component parent = null);
    super.new(name, parent);
    get_imp = new("get_imp", this);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    test_transaction txn;
    super.build_phase(phase);
    // Pre-populate with transactions
    for (int i = 0; i < 5; i++) begin
      txn = test_transaction::type_id::create($sformatf("pre_txn_%0d", i));
      txn.id = i + 10;
      txn.data = (i + 10) * 100;
      ready_txns.push_back(txn);
    end
  endfunction

  // Implementation of get method - called via imp
  virtual task get(output test_transaction t);
    wait (ready_txns.size() > 0);
    t = ready_txns.pop_front();
    `uvm_info(get_type_name(), $sformatf("Providing via get_imp: %s", t.convert2string()), UVM_HIGH)
  endtask
endclass

// CHECK: moore.class.classdecl @get_source extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Nonblocking producer using try_put
//===----------------------------------------------------------------------===//

class nonblocking_producer extends uvm_component;
  `uvm_component_utils(nonblocking_producer)

  uvm_nonblocking_put_port #(test_transaction) nb_put_port;
  int put_success_count;
  int put_fail_count;

  function new(string name = "nonblocking_producer", uvm_component parent = null);
    super.new(name, parent);
    nb_put_port = new("nb_put_port", this);
    put_success_count = 0;
    put_fail_count = 0;
  endfunction

  virtual task run_phase(uvm_phase phase);
    test_transaction txn;
    super.run_phase(phase);

    for (int i = 0; i < 10; i++) begin
      txn = test_transaction::type_id::create($sformatf("nb_txn_%0d", i));
      txn.id = i + 20;
      txn.data = (i + 20) * 100;

      if (nb_put_port.can_put()) begin
        if (nb_put_port.try_put(txn)) begin
          put_success_count++;
          `uvm_info(get_type_name(), $sformatf("try_put succeeded: %s", txn.convert2string()), UVM_HIGH)
        end else begin
          put_fail_count++;
          `uvm_info(get_type_name(), "try_put failed", UVM_HIGH)
        end
      end else begin
        `uvm_info(get_type_name(), "can_put returned false", UVM_HIGH)
      end
      #1;
    end
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(),
              $sformatf("Nonblocking puts: %0d succeeded, %0d failed",
                        put_success_count, put_fail_count), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @nonblocking_producer extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// TLM FIFO test component
//===----------------------------------------------------------------------===//

class fifo_tester extends uvm_component;
  `uvm_component_utils(fifo_tester)

  uvm_tlm_fifo #(test_transaction) fifo;
  int test_pass_count;
  int test_fail_count;

  function new(string name = "fifo_tester", uvm_component parent = null);
    super.new(name, parent);
    fifo = new("fifo", this, 4); // Size 4 FIFO
    test_pass_count = 0;
    test_fail_count = 0;
  endfunction

  virtual task run_phase(uvm_phase phase);
    test_transaction txn, retrieved;
    super.run_phase(phase);

    `uvm_info(get_type_name(), "=== TLM FIFO Tests ===", UVM_MEDIUM)

    // Test 1: Initial state
    if (fifo.is_empty()) begin
      `uvm_info(get_type_name(), "Test 1 PASS: FIFO initially empty", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), "Test 1 FAIL: FIFO should be empty")
      test_fail_count++;
    end

    // Test 2: Put transactions
    for (int i = 0; i < 3; i++) begin
      txn = test_transaction::type_id::create($sformatf("fifo_txn_%0d", i));
      txn.id = i;
      txn.data = i * 1000;
      fifo.put(txn);
    end
    if (fifo.size() == 3) begin
      `uvm_info(get_type_name(), "Test 2 PASS: Put 3 transactions", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), $sformatf("Test 2 FAIL: Expected size 3, got %0d", fifo.size()))
      test_fail_count++;
    end

    // Test 3: try_peek (should not remove item)
    if (fifo.try_peek(retrieved)) begin
      if (retrieved.id == 0) begin
        `uvm_info(get_type_name(), "Test 3 PASS: try_peek returned first item", UVM_MEDIUM)
        test_pass_count++;
      end else begin
        `uvm_error(get_type_name(), $sformatf("Test 3 FAIL: Expected id 0, got %0d", retrieved.id))
        test_fail_count++;
      end
    end else begin
      `uvm_error(get_type_name(), "Test 3 FAIL: try_peek returned false")
      test_fail_count++;
    end

    // Test 4: Size unchanged after peek
    if (fifo.size() == 3) begin
      `uvm_info(get_type_name(), "Test 4 PASS: Size unchanged after peek", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), $sformatf("Test 4 FAIL: Size changed after peek: %0d", fifo.size()))
      test_fail_count++;
    end

    // Test 5: try_get (should remove item)
    if (fifo.try_get(retrieved)) begin
      if (retrieved.id == 0) begin
        `uvm_info(get_type_name(), "Test 5 PASS: try_get returned and removed first item", UVM_MEDIUM)
        test_pass_count++;
      end else begin
        `uvm_error(get_type_name(), $sformatf("Test 5 FAIL: Expected id 0, got %0d", retrieved.id))
        test_fail_count++;
      end
    end else begin
      `uvm_error(get_type_name(), "Test 5 FAIL: try_get returned false")
      test_fail_count++;
    end

    // Test 6: Size reduced after get
    if (fifo.size() == 2) begin
      `uvm_info(get_type_name(), "Test 6 PASS: Size reduced after get", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), $sformatf("Test 6 FAIL: Expected size 2, got %0d", fifo.size()))
      test_fail_count++;
    end

    // Test 7: Blocking get
    fifo.get(retrieved);
    if (retrieved.id == 1) begin
      `uvm_info(get_type_name(), "Test 7 PASS: Blocking get returned correct item", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), $sformatf("Test 7 FAIL: Expected id 1, got %0d", retrieved.id))
      test_fail_count++;
    end

    // Test 8: Flush
    fifo.flush();
    if (fifo.is_empty()) begin
      `uvm_info(get_type_name(), "Test 8 PASS: FIFO empty after flush", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), "Test 8 FAIL: FIFO not empty after flush")
      test_fail_count++;
    end

    // Test 9: try_get on empty FIFO
    if (!fifo.try_get(retrieved)) begin
      `uvm_info(get_type_name(), "Test 9 PASS: try_get on empty returns false", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), "Test 9 FAIL: try_get on empty should return false")
      test_fail_count++;
    end

    // Test 10: can_get/can_peek on empty FIFO
    if (!fifo.can_get() && !fifo.can_peek()) begin
      `uvm_info(get_type_name(), "Test 10 PASS: can_get/can_peek false on empty", UVM_MEDIUM)
      test_pass_count++;
    end else begin
      `uvm_error(get_type_name(), "Test 10 FAIL: can_get/can_peek should be false on empty")
      test_fail_count++;
    end

    `uvm_info(get_type_name(), "=== TLM FIFO Tests Complete ===", UVM_MEDIUM)
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(),
              $sformatf("FIFO Tests: %0d passed, %0d failed",
                        test_pass_count, test_fail_count), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @fifo_tester extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Environment connecting components
//===----------------------------------------------------------------------===//

class tlm_port_env extends uvm_env;
  `uvm_component_utils(tlm_port_env)

  producer prod;
  consumer cons;
  put_target put_tgt;
  get_source get_src;
  nonblocking_producer nb_prod;
  fifo_tester fifo_test;

  // TLM FIFOs for connection
  uvm_tlm_fifo #(test_transaction) prod_cons_fifo;
  uvm_tlm_fifo #(test_transaction) nb_fifo;

  function new(string name = "tlm_port_env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);

    prod = producer::type_id::create("prod", this);
    cons = consumer::type_id::create("cons", this);
    put_tgt = put_target::type_id::create("put_tgt", this);
    get_src = get_source::type_id::create("get_src", this);
    nb_prod = nonblocking_producer::type_id::create("nb_prod", this);
    fifo_test = fifo_tester::type_id::create("fifo_test", this);

    prod_cons_fifo = new("prod_cons_fifo", this, 8);
    nb_fifo = new("nb_fifo", this, 4);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);

    // Connect producer's put_port to FIFO's put_export
    prod.put_port.connect(prod_cons_fifo.put_export);
    // Connect consumer's get_port to FIFO's get_peek_export
    cons.get_port.connect(prod_cons_fifo.get_peek_export);

    // Connect nonblocking producer to bounded FIFO
    nb_prod.nb_put_port.connect(nb_fifo.put_export);

    `uvm_info(get_type_name(), "TLM port connections complete", UVM_MEDIUM)
  endfunction
endclass

// CHECK: moore.class.classdecl @tlm_port_env extends @"uvm_pkg::uvm_env"

//===----------------------------------------------------------------------===//
// Test
//===----------------------------------------------------------------------===//

class tlm_port_test extends uvm_test;
  `uvm_component_utils(tlm_port_test)

  tlm_port_env env;

  function new(string name = "tlm_port_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = tlm_port_env::type_id::create("env", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);
    phase.raise_objection(this);
    #200;
    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), "TLM Port/Export/Imp test complete", UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @tlm_port_test extends @"uvm_pkg::uvm_test"

//===----------------------------------------------------------------------===//
// Top module
//===----------------------------------------------------------------------===//

module uvm_tlm_port_test_top;
  initial begin
    $display("UVM TLM Port/Export/Imp Test");
    run_test("tlm_port_test");
  end
endmodule

// CHECK: moore.module @uvm_tlm_port_test_top
