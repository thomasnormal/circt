//===----------------------------------------------------------------------===//
// UVM TLM FIFO Pattern Test
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
//
// This test validates UVM TLM FIFO patterns including:
// 1. uvm_tlm_fifo basic operations (put, get, peek)
// 2. uvm_tlm_analysis_fifo
// 3. try_put, try_get, try_peek (nonblocking)
// 4. can_put, can_get, can_peek (availability checks)
// 5. FIFO size and fullness (size, used, is_empty, is_full)
// 6. flush operation
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`include "uvm_macros.svh"

package tlm_fifo_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Transaction class for FIFO testing
  //==========================================================================
  class fifo_transaction extends uvm_sequence_item;
    `uvm_object_utils(fifo_transaction)

    rand bit [31:0] data;
    rand bit [15:0] addr;
    rand bit [7:0]  id;
    rand bit        write;

    function new(string name = "fifo_transaction");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("id=%0d addr=0x%04h data=0x%08h write=%b",
                       id, addr, data, write);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      fifo_transaction rhs_tx;
      super.do_copy(rhs);
      if ($cast(rhs_tx, rhs)) begin
        data = rhs_tx.data;
        addr = rhs_tx.addr;
        id = rhs_tx.id;
        write = rhs_tx.write;
      end
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      fifo_transaction rhs_tx;
      if (!$cast(rhs_tx, rhs))
        return 0;
      return (data == rhs_tx.data) && (addr == rhs_tx.addr) &&
             (id == rhs_tx.id) && (write == rhs_tx.write);
    endfunction

  endclass

  //==========================================================================
  // Test 1: Basic uvm_tlm_fifo operations (put, get, peek)
  //==========================================================================
  class basic_fifo_test extends uvm_component;
    `uvm_component_utils(basic_fifo_test)

    uvm_tlm_fifo #(fifo_transaction) fifo;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create FIFO with max size of 8
      fifo = new("basic_fifo", this, 8);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 1: Basic FIFO Operations ===", UVM_LOW)

      // Test 1.1: Blocking put
      `uvm_info(get_type_name(), "Test 1.1: Blocking put", UVM_MEDIUM)
      for (int i = 0; i < 5; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("txn_%0d", i));
        txn.id = i;
        txn.data = i * 1000;
        txn.addr = i * 4;
        txn.write = (i % 2 == 0);
        fifo.put(txn);
        `uvm_info(get_type_name(),
                  $sformatf("  Put transaction: %s", txn.convert2string()), UVM_HIGH)
      end
      if (fifo.size() == 5) begin
        `uvm_info(get_type_name(), "  PASS: Put 5 transactions successfully", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected size 5, got %0d", fifo.size()))
        fail_count++;
      end

      // Test 1.2: Blocking peek (should not remove item)
      `uvm_info(get_type_name(), "Test 1.2: Blocking peek", UVM_MEDIUM)
      fifo.peek(retrieved);
      if (retrieved.id == 0 && fifo.size() == 5) begin
        `uvm_info(get_type_name(),
                  $sformatf("  PASS: Peek returned first item without removal: %s",
                            retrieved.convert2string()), UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: Peek behavior incorrect")
        fail_count++;
      end

      // Test 1.3: Blocking get (should remove item)
      `uvm_info(get_type_name(), "Test 1.3: Blocking get", UVM_MEDIUM)
      for (int i = 0; i < 3; i++) begin
        fifo.get(retrieved);
        if (retrieved.id == i) begin
          `uvm_info(get_type_name(),
                    $sformatf("  Got transaction %0d: %s", i, retrieved.convert2string()),
                    UVM_HIGH)
        end else begin
          `uvm_error(get_type_name(),
                     $sformatf("  FAIL: Expected id %0d, got %0d", i, retrieved.id))
          fail_count++;
        end
      end
      if (fifo.size() == 2) begin
        `uvm_info(get_type_name(), "  PASS: Got 3 transactions, 2 remaining", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected size 2, got %0d", fifo.size()))
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Basic FIFO Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Test 2: uvm_tlm_analysis_fifo
  //==========================================================================
  class analysis_fifo_test extends uvm_component;
    `uvm_component_utils(analysis_fifo_test)

    uvm_tlm_analysis_fifo #(fifo_transaction) analysis_fifo;
    uvm_analysis_port #(fifo_transaction) analysis_port;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Analysis FIFO is unbounded by default
      analysis_fifo = new("analysis_fifo", this);
      analysis_port = new("analysis_port", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      // Connect analysis port to analysis FIFO's analysis_export
      analysis_port.connect(analysis_fifo.analysis_export);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 2: Analysis FIFO Operations ===", UVM_LOW)

      // Test 2.1: Write via analysis port
      `uvm_info(get_type_name(), "Test 2.1: Write via analysis_port", UVM_MEDIUM)
      for (int i = 0; i < 10; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("analysis_txn_%0d", i));
        txn.id = i + 100;
        txn.data = 32'hDEAD0000 + i;
        txn.addr = 16'hBEEF;
        txn.write = 1;
        analysis_port.write(txn);
      end
      if (analysis_fifo.size() == 10) begin
        `uvm_info(get_type_name(),
                  "  PASS: Analysis port wrote 10 transactions", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected size 10, got %0d", analysis_fifo.size()))
        fail_count++;
      end

      // Test 2.2: Analysis FIFO is unbounded
      `uvm_info(get_type_name(), "Test 2.2: Verify unbounded nature", UVM_MEDIUM)
      if (!analysis_fifo.is_full()) begin
        `uvm_info(get_type_name(),
                  "  PASS: Analysis FIFO reports not full (unbounded)", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: Analysis FIFO should be unbounded")
        fail_count++;
      end

      // Test 2.3: Get from analysis FIFO
      `uvm_info(get_type_name(), "Test 2.3: Get from analysis FIFO", UVM_MEDIUM)
      analysis_fifo.get(retrieved);
      if (retrieved.id == 100 && analysis_fifo.size() == 9) begin
        `uvm_info(get_type_name(),
                  $sformatf("  PASS: Got first transaction: %s", retrieved.convert2string()),
                  UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: Get operation incorrect")
        fail_count++;
      end

      // Test 2.4: Direct write method
      `uvm_info(get_type_name(), "Test 2.4: Direct write method", UVM_MEDIUM)
      txn = fifo_transaction::type_id::create("direct_write_txn");
      txn.id = 200;
      txn.data = 32'hCAFEBABE;
      analysis_fifo.write(txn);
      if (analysis_fifo.size() == 10) begin
        `uvm_info(get_type_name(),
                  "  PASS: Direct write added transaction", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected size 10, got %0d", analysis_fifo.size()))
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Analysis FIFO Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Test 3: Nonblocking operations (try_put, try_get, try_peek)
  //==========================================================================
  class nonblocking_fifo_test extends uvm_component;
    `uvm_component_utils(nonblocking_fifo_test)

    uvm_tlm_fifo #(fifo_transaction) fifo;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create bounded FIFO with size 4
      fifo = new("bounded_fifo", this, 4);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;
      bit success;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 3: Nonblocking Operations ===", UVM_LOW)

      // Test 3.1: try_put on empty FIFO
      `uvm_info(get_type_name(), "Test 3.1: try_put on empty FIFO", UVM_MEDIUM)
      txn = fifo_transaction::type_id::create("try_put_txn");
      txn.id = 50;
      txn.data = 32'h12345678;
      success = fifo.try_put(txn);
      if (success && fifo.size() == 1) begin
        `uvm_info(get_type_name(), "  PASS: try_put succeeded on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_put should succeed on empty FIFO")
        fail_count++;
      end

      // Test 3.2: try_put until full
      `uvm_info(get_type_name(), "Test 3.2: try_put until full", UVM_MEDIUM)
      for (int i = 0; i < 3; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("fill_txn_%0d", i));
        txn.id = 51 + i;
        success = fifo.try_put(txn);
        if (!success) begin
          `uvm_error(get_type_name(),
                     $sformatf("  FAIL: try_put %0d should succeed", i))
          fail_count++;
        end
      end
      if (fifo.is_full() && fifo.size() == 4) begin
        `uvm_info(get_type_name(), "  PASS: FIFO is now full", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFO should be full")
        fail_count++;
      end

      // Test 3.3: try_put on full FIFO (should fail)
      `uvm_info(get_type_name(), "Test 3.3: try_put on full FIFO", UVM_MEDIUM)
      txn = fifo_transaction::type_id::create("overflow_txn");
      txn.id = 99;
      success = fifo.try_put(txn);
      if (!success) begin
        `uvm_info(get_type_name(), "  PASS: try_put correctly failed on full FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_put should fail on full FIFO")
        fail_count++;
      end

      // Test 3.4: try_peek
      `uvm_info(get_type_name(), "Test 3.4: try_peek", UVM_MEDIUM)
      success = fifo.try_peek(retrieved);
      if (success && retrieved.id == 50 && fifo.size() == 4) begin
        `uvm_info(get_type_name(),
                  $sformatf("  PASS: try_peek returned first item: %s",
                            retrieved.convert2string()), UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_peek behavior incorrect")
        fail_count++;
      end

      // Test 3.5: try_get
      `uvm_info(get_type_name(), "Test 3.5: try_get", UVM_MEDIUM)
      success = fifo.try_get(retrieved);
      if (success && retrieved.id == 50 && fifo.size() == 3) begin
        `uvm_info(get_type_name(),
                  $sformatf("  PASS: try_get returned and removed first item: %s",
                            retrieved.convert2string()), UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_get behavior incorrect")
        fail_count++;
      end

      // Test 3.6: Empty the FIFO
      `uvm_info(get_type_name(), "Test 3.6: Empty the FIFO with try_get", UVM_MEDIUM)
      while (fifo.try_get(retrieved)) begin
        `uvm_info(get_type_name(),
                  $sformatf("  Got: %s", retrieved.convert2string()), UVM_HIGH)
      end
      if (fifo.is_empty()) begin
        `uvm_info(get_type_name(), "  PASS: FIFO is now empty", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFO should be empty")
        fail_count++;
      end

      // Test 3.7: try_get on empty FIFO (should fail)
      `uvm_info(get_type_name(), "Test 3.7: try_get on empty FIFO", UVM_MEDIUM)
      success = fifo.try_get(retrieved);
      if (!success) begin
        `uvm_info(get_type_name(), "  PASS: try_get correctly failed on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_get should fail on empty FIFO")
        fail_count++;
      end

      // Test 3.8: try_peek on empty FIFO (should fail)
      `uvm_info(get_type_name(), "Test 3.8: try_peek on empty FIFO", UVM_MEDIUM)
      success = fifo.try_peek(retrieved);
      if (!success) begin
        `uvm_info(get_type_name(), "  PASS: try_peek correctly failed on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: try_peek should fail on empty FIFO")
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Nonblocking FIFO Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Test 4: Availability checks (can_put, can_get, can_peek)
  //==========================================================================
  class availability_check_test extends uvm_component;
    `uvm_component_utils(availability_check_test)

    uvm_tlm_fifo #(fifo_transaction) fifo;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create bounded FIFO with size 3
      fifo = new("check_fifo", this, 3);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 4: Availability Checks ===", UVM_LOW)

      // Test 4.1: can_put on empty FIFO
      `uvm_info(get_type_name(), "Test 4.1: can_put on empty FIFO", UVM_MEDIUM)
      if (fifo.can_put()) begin
        `uvm_info(get_type_name(), "  PASS: can_put returns true on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_put should return true on empty FIFO")
        fail_count++;
      end

      // Test 4.2: can_get on empty FIFO
      `uvm_info(get_type_name(), "Test 4.2: can_get on empty FIFO", UVM_MEDIUM)
      if (!fifo.can_get()) begin
        `uvm_info(get_type_name(), "  PASS: can_get returns false on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_get should return false on empty FIFO")
        fail_count++;
      end

      // Test 4.3: can_peek on empty FIFO
      `uvm_info(get_type_name(), "Test 4.3: can_peek on empty FIFO", UVM_MEDIUM)
      if (!fifo.can_peek()) begin
        `uvm_info(get_type_name(), "  PASS: can_peek returns false on empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_peek should return false on empty FIFO")
        fail_count++;
      end

      // Fill the FIFO
      for (int i = 0; i < 3; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("fill_txn_%0d", i));
        txn.id = i;
        fifo.put(txn);
      end

      // Test 4.4: can_put on full FIFO
      `uvm_info(get_type_name(), "Test 4.4: can_put on full FIFO", UVM_MEDIUM)
      if (!fifo.can_put()) begin
        `uvm_info(get_type_name(), "  PASS: can_put returns false on full FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_put should return false on full FIFO")
        fail_count++;
      end

      // Test 4.5: can_get on non-empty FIFO
      `uvm_info(get_type_name(), "Test 4.5: can_get on non-empty FIFO", UVM_MEDIUM)
      if (fifo.can_get()) begin
        `uvm_info(get_type_name(), "  PASS: can_get returns true on non-empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_get should return true on non-empty FIFO")
        fail_count++;
      end

      // Test 4.6: can_peek on non-empty FIFO
      `uvm_info(get_type_name(), "Test 4.6: can_peek on non-empty FIFO", UVM_MEDIUM)
      if (fifo.can_peek()) begin
        `uvm_info(get_type_name(), "  PASS: can_peek returns true on non-empty FIFO", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_peek should return true on non-empty FIFO")
        fail_count++;
      end

      // Test 4.7: can_put after removing one item
      `uvm_info(get_type_name(), "Test 4.7: can_put after removing one item", UVM_MEDIUM)
      fifo.get(retrieved);
      if (fifo.can_put()) begin
        `uvm_info(get_type_name(),
                  "  PASS: can_put returns true after removing item", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   "  FAIL: can_put should return true after removing item")
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Availability Check Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Test 5: FIFO size and fullness (size, used, is_empty, is_full)
  //==========================================================================
  class size_fullness_test extends uvm_component;
    `uvm_component_utils(size_fullness_test)

    uvm_tlm_fifo #(fifo_transaction) bounded_fifo;
    uvm_tlm_fifo #(fifo_transaction) unbounded_fifo;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create bounded FIFO with size 5
      bounded_fifo = new("bounded_fifo", this, 5);
      // Create unbounded FIFO (size 0 means unbounded)
      unbounded_fifo = new("unbounded_fifo", this, 0);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 5: Size and Fullness ===", UVM_LOW)

      // Test 5.1: Initial state - is_empty
      `uvm_info(get_type_name(), "Test 5.1: Initial state is_empty", UVM_MEDIUM)
      if (bounded_fifo.is_empty() && unbounded_fifo.is_empty()) begin
        `uvm_info(get_type_name(), "  PASS: Both FIFOs initially empty", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFOs should be initially empty")
        fail_count++;
      end

      // Test 5.2: Initial size is 0
      `uvm_info(get_type_name(), "Test 5.2: Initial size is 0", UVM_MEDIUM)
      if (bounded_fifo.size() == 0 && unbounded_fifo.size() == 0) begin
        `uvm_info(get_type_name(), "  PASS: Both FIFOs have size 0", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFOs should have size 0")
        fail_count++;
      end

      // Test 5.3: Initial used is 0
      `uvm_info(get_type_name(), "Test 5.3: Initial used is 0", UVM_MEDIUM)
      if (bounded_fifo.used() == 0 && unbounded_fifo.used() == 0) begin
        `uvm_info(get_type_name(), "  PASS: Both FIFOs have used 0", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFOs should have used 0")
        fail_count++;
      end

      // Test 5.4: is_full on bounded FIFO after filling
      `uvm_info(get_type_name(), "Test 5.4: is_full on bounded FIFO", UVM_MEDIUM)
      for (int i = 0; i < 5; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("bounded_txn_%0d", i));
        txn.id = i;
        bounded_fifo.put(txn);
      end
      if (bounded_fifo.is_full() && bounded_fifo.size() == 5 && bounded_fifo.used() == 5) begin
        `uvm_info(get_type_name(),
                  "  PASS: Bounded FIFO is full, size=5, used=5", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected full with size 5, got size=%0d used=%0d is_full=%b",
                             bounded_fifo.size(), bounded_fifo.used(), bounded_fifo.is_full()))
        fail_count++;
      end

      // Test 5.5: is_full on unbounded FIFO (should never be full)
      `uvm_info(get_type_name(), "Test 5.5: is_full on unbounded FIFO", UVM_MEDIUM)
      for (int i = 0; i < 100; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("unbounded_txn_%0d", i));
        txn.id = i;
        unbounded_fifo.put(txn);
      end
      if (!unbounded_fifo.is_full() && unbounded_fifo.size() == 100) begin
        `uvm_info(get_type_name(),
                  "  PASS: Unbounded FIFO is not full with 100 items", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: Unbounded FIFO should never be full")
        fail_count++;
      end

      // Test 5.6: is_empty transitions
      `uvm_info(get_type_name(), "Test 5.6: is_empty transitions", UVM_MEDIUM)
      // Empty the bounded FIFO
      while (!bounded_fifo.is_empty()) begin
        bounded_fifo.get(retrieved);
      end
      if (bounded_fifo.is_empty() && !bounded_fifo.is_full()) begin
        `uvm_info(get_type_name(),
                  "  PASS: Bounded FIFO is empty and not full after draining", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: Bounded FIFO should be empty after draining")
        fail_count++;
      end

      // Test 5.7: size and used should be equal
      `uvm_info(get_type_name(), "Test 5.7: size equals used", UVM_MEDIUM)
      // Put 3 items in bounded FIFO
      for (int i = 0; i < 3; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("size_test_txn_%0d", i));
        txn.id = i;
        bounded_fifo.put(txn);
      end
      if (bounded_fifo.size() == bounded_fifo.used() && bounded_fifo.size() == 3) begin
        `uvm_info(get_type_name(), "  PASS: size() == used() == 3", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: size=%0d, used=%0d, expected 3",
                             bounded_fifo.size(), bounded_fifo.used()))
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Size/Fullness Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Test 6: Flush operation
  //==========================================================================
  class flush_test extends uvm_component;
    `uvm_component_utils(flush_test)

    uvm_tlm_fifo #(fifo_transaction) fifo;
    int pass_count;
    int fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      pass_count = 0;
      fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("flush_fifo", this, 10);
    endfunction

    virtual task run_phase(uvm_phase phase);
      fifo_transaction txn, retrieved;
      bit success;

      super.run_phase(phase);

      `uvm_info(get_type_name(), "=== Test 6: Flush Operation ===", UVM_LOW)

      // Test 6.1: Flush empty FIFO (should be safe)
      `uvm_info(get_type_name(), "Test 6.1: Flush empty FIFO", UVM_MEDIUM)
      fifo.flush();
      if (fifo.is_empty() && fifo.size() == 0) begin
        `uvm_info(get_type_name(), "  PASS: Flush on empty FIFO is safe", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFO should remain empty after flush")
        fail_count++;
      end

      // Test 6.2: Flush after adding items
      `uvm_info(get_type_name(), "Test 6.2: Flush after adding items", UVM_MEDIUM)
      for (int i = 0; i < 7; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("flush_txn_%0d", i));
        txn.id = i;
        txn.data = 32'hFFFFFFFF - i;
        fifo.put(txn);
      end
      if (fifo.size() == 7) begin
        `uvm_info(get_type_name(), "  Added 7 transactions", UVM_HIGH)
      end
      fifo.flush();
      if (fifo.is_empty() && fifo.size() == 0) begin
        `uvm_info(get_type_name(), "  PASS: Flush cleared all items", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("  FAIL: Expected empty after flush, got size=%0d", fifo.size()))
        fail_count++;
      end

      // Test 6.3: Verify operations work after flush
      `uvm_info(get_type_name(), "Test 6.3: Operations after flush", UVM_MEDIUM)
      txn = fifo_transaction::type_id::create("post_flush_txn");
      txn.id = 99;
      txn.data = 32'hABCD1234;
      fifo.put(txn);
      success = fifo.try_get(retrieved);
      if (success && retrieved.id == 99 && retrieved.data == 32'hABCD1234) begin
        `uvm_info(get_type_name(),
                  "  PASS: FIFO operations work correctly after flush", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: FIFO operations broken after flush")
        fail_count++;
      end

      // Test 6.4: can_put after flush
      `uvm_info(get_type_name(), "Test 6.4: can_put after flush", UVM_MEDIUM)
      // Fill the FIFO
      for (int i = 0; i < 10; i++) begin
        txn = fifo_transaction::type_id::create($sformatf("fill_txn_%0d", i));
        txn.id = i;
        fifo.put(txn);
      end
      if (!fifo.can_put()) begin
        `uvm_info(get_type_name(), "  FIFO is full, can_put returns false", UVM_HIGH)
      end
      fifo.flush();
      if (fifo.can_put()) begin
        `uvm_info(get_type_name(), "  PASS: can_put returns true after flush", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_put should return true after flush")
        fail_count++;
      end

      // Test 6.5: can_get after flush
      `uvm_info(get_type_name(), "Test 6.5: can_get after flush", UVM_MEDIUM)
      if (!fifo.can_get()) begin
        `uvm_info(get_type_name(), "  PASS: can_get returns false after flush", UVM_MEDIUM)
        pass_count++;
      end else begin
        `uvm_error(get_type_name(), "  FAIL: can_get should return false after flush")
        fail_count++;
      end

    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(),
                $sformatf("Flush Test Results: %0d passed, %0d failed",
                          pass_count, fail_count), UVM_LOW)
    endfunction

  endclass

  //==========================================================================
  // Master environment containing all test components
  //==========================================================================
  class tlm_fifo_test_env extends uvm_env;
    `uvm_component_utils(tlm_fifo_test_env)

    basic_fifo_test        test1_basic;
    analysis_fifo_test     test2_analysis;
    nonblocking_fifo_test  test3_nonblocking;
    availability_check_test test4_availability;
    size_fullness_test     test5_size;
    flush_test             test6_flush;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      test1_basic        = basic_fifo_test::type_id::create("test1_basic", this);
      test2_analysis     = analysis_fifo_test::type_id::create("test2_analysis", this);
      test3_nonblocking  = nonblocking_fifo_test::type_id::create("test3_nonblocking", this);
      test4_availability = availability_check_test::type_id::create("test4_availability", this);
      test5_size         = size_fullness_test::type_id::create("test5_size", this);
      test6_flush        = flush_test::type_id::create("test6_flush", this);
    endfunction

  endclass

  //==========================================================================
  // Top-level test
  //==========================================================================
  class tlm_fifo_test extends uvm_test;
    `uvm_component_utils(tlm_fifo_test)

    tlm_fifo_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = tlm_fifo_test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      super.run_phase(phase);
      phase.raise_objection(this, "Starting TLM FIFO tests");
      // Allow time for all tests to run
      #1000;
      phase.drop_objection(this, "TLM FIFO tests complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info(get_type_name(), "", UVM_LOW)
      `uvm_info(get_type_name(), "============================================", UVM_LOW)
      `uvm_info(get_type_name(), "  UVM TLM FIFO Pattern Test Complete", UVM_LOW)
      `uvm_info(get_type_name(), "============================================", UVM_LOW)
      `uvm_info(get_type_name(), "", UVM_LOW)
      `uvm_info(get_type_name(), "Tested patterns:", UVM_LOW)
      `uvm_info(get_type_name(), "  1. Basic FIFO operations (put, get, peek)", UVM_LOW)
      `uvm_info(get_type_name(), "  2. Analysis FIFO with analysis_export", UVM_LOW)
      `uvm_info(get_type_name(), "  3. Nonblocking operations (try_put/get/peek)", UVM_LOW)
      `uvm_info(get_type_name(), "  4. Availability checks (can_put/get/peek)", UVM_LOW)
      `uvm_info(get_type_name(), "  5. Size and fullness (size, used, is_empty, is_full)", UVM_LOW)
      `uvm_info(get_type_name(), "  6. Flush operation", UVM_LOW)
      `uvm_info(get_type_name(), "", UVM_LOW)
    endfunction

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import tlm_fifo_test_pkg::*;

  initial begin
    `uvm_info("TB_TOP", "Starting UVM TLM FIFO Pattern Test", UVM_NONE)
    run_test("tlm_fifo_test");
  end

endmodule
