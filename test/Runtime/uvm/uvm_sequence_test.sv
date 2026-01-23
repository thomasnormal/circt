//===----------------------------------------------------------------------===//
// UVM Sequence Patterns Test
//===----------------------------------------------------------------------===//
// Comprehensive test for UVM sequence patterns including:
// 1. uvm_sequence_item with constraints
// 2. uvm_sequence with body() task
// 3. Sequence nesting (sub-sequences)
// 4. start_item/finish_item pattern
// 5. get_response pattern
// 6. Virtual sequences
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package sequence_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // SECTION 1: uvm_sequence_item with constraints
  //==========================================================================

  // Basic sequence item with various constraint types
  class constrained_item extends uvm_sequence_item;
    `uvm_object_utils(constrained_item)

    // Randomizable fields
    rand bit [7:0] data;
    rand bit [15:0] addr;
    rand bit [3:0] burst_len;
    rand bit write;
    rand bit [1:0] size;

    // Non-random fields
    int transaction_id;
    time timestamp;

    // Simple range constraint
    constraint c_data_range { data inside {[8'h10:8'hF0]}; }

    // Address alignment constraint
    constraint c_addr_align { addr[1:0] == 2'b00; }

    // Conditional constraint - burst length depends on write type
    constraint c_burst_len {
      write -> burst_len inside {[1:8]};
      !write -> burst_len inside {[1:4]};
    }

    // Size constraint
    constraint c_size { size inside {0, 1, 2}; }

    // Soft constraint (can be overridden)
    constraint c_default_write { soft write == 1'b1; }

    function new(string name = "constrained_item");
      super.new(name);
      transaction_id = -1;
    endfunction

    virtual function string convert2string();
      return $sformatf("id=%0d %s addr=0x%04h data=0x%02h burst=%0d size=%0d",
                       transaction_id,
                       write ? "WR" : "RD",
                       addr, data, burst_len, size);
    endfunction

    // Clone method
    virtual function uvm_object clone();
      constrained_item cp;
      cp = new();
      cp.data = this.data;
      cp.addr = this.addr;
      cp.burst_len = this.burst_len;
      cp.write = this.write;
      cp.size = this.size;
      cp.transaction_id = this.transaction_id;
      cp.timestamp = this.timestamp;
      return cp;
    endfunction

    // Compare method
    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      constrained_item rhs_item;
      if (!$cast(rhs_item, rhs))
        return 0;
      return (this.data == rhs_item.data) &&
             (this.addr == rhs_item.addr) &&
             (this.burst_len == rhs_item.burst_len) &&
             (this.write == rhs_item.write);
    endfunction

  endclass

  // Response item for request-response pattern
  class response_item extends uvm_sequence_item;
    `uvm_object_utils(response_item)

    bit [7:0] data;
    bit [1:0] status;  // 0=OK, 1=ERROR, 2=RETRY
    int req_transaction_id;

    function new(string name = "response_item");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("RSP id=%0d data=0x%02h status=%0d",
                       req_transaction_id, data, status);
    endfunction
  endclass

  //==========================================================================
  // SECTION 2: Basic uvm_sequence with body() task
  //==========================================================================

  class basic_sequence extends uvm_sequence #(constrained_item);
    `uvm_object_utils(basic_sequence)

    int num_items = 5;
    int seq_id = 0;

    // Track transaction IDs
    protected int m_next_transaction_id = 0;

    function new(string name = "basic_sequence");
      super.new(name);
    endfunction

    // The body() task is the main entry point for sequence execution
    virtual task body();
      constrained_item item;

      `uvm_info("BASIC_SEQ", $sformatf("Starting basic sequence (id=%0d, items=%0d)",
                                       seq_id, num_items), UVM_MEDIUM)

      repeat (num_items) begin
        item = constrained_item::type_id::create("item");

        // Standard start_item/finish_item pattern
        start_item(item);

        // Randomize after start_item (sequencer has granted access)
        if (!item.randomize())
          `uvm_error("BASIC_SEQ", "Randomization failed")

        item.transaction_id = seq_id * 100 + m_next_transaction_id++;
        item.timestamp = $time;

        finish_item(item);

        `uvm_info("BASIC_SEQ", $sformatf("Sent: %s", item.convert2string()), UVM_HIGH)
      end

      `uvm_info("BASIC_SEQ", $sformatf("Completed basic sequence (id=%0d)", seq_id), UVM_MEDIUM)
    endtask

  endclass

  //==========================================================================
  // SECTION 3: Sequence nesting (sub-sequences)
  //==========================================================================

  // Inner sequence - represents a burst transaction
  class burst_sequence extends uvm_sequence #(constrained_item);
    `uvm_object_utils(burst_sequence)

    rand int unsigned burst_count;
    bit [15:0] base_addr;

    constraint c_burst_count { burst_count inside {[2:8]}; }

    function new(string name = "burst_sequence");
      super.new(name);
      base_addr = 16'h1000;
    endfunction

    virtual task body();
      constrained_item item;

      `uvm_info("BURST_SEQ", $sformatf("Starting burst of %0d items at addr 0x%04h",
                                       burst_count, base_addr), UVM_MEDIUM)

      for (int i = 0; i < burst_count; i++) begin
        item = constrained_item::type_id::create("item");

        start_item(item);
        if (!item.randomize() with {
          addr == base_addr + (i * 4);  // Sequential addresses
          write == 1'b1;                // All writes
        })
          `uvm_error("BURST_SEQ", "Randomization failed")
        finish_item(item);
      end

      `uvm_info("BURST_SEQ", "Burst sequence complete", UVM_MEDIUM)
    endtask
  endclass

  // Outer sequence - contains nested sub-sequences
  class nested_sequence extends uvm_sequence #(constrained_item);
    `uvm_object_utils(nested_sequence)

    int num_bursts = 3;

    function new(string name = "nested_sequence");
      super.new(name);
    endfunction

    virtual task body();
      burst_sequence burst_seq;
      constrained_item single_item;

      `uvm_info("NESTED_SEQ", $sformatf("Starting nested sequence with %0d bursts",
                                        num_bursts), UVM_MEDIUM)

      // Execute multiple sub-sequences
      for (int b = 0; b < num_bursts; b++) begin
        // Create and start a sub-sequence
        burst_seq = burst_sequence::type_id::create($sformatf("burst_%0d", b));
        burst_seq.base_addr = 16'h1000 + (b * 16'h100);

        // Randomize the sub-sequence's count
        if (!burst_seq.randomize())
          `uvm_error("NESTED_SEQ", "Sub-sequence randomization failed")

        // Start sub-sequence on same sequencer
        // The start() method runs the sub-sequence's body() task
        `uvm_info("NESTED_SEQ", $sformatf("Starting burst %0d (count=%0d, addr=0x%04h)",
                                          b, burst_seq.burst_count, burst_seq.base_addr), UVM_MEDIUM)
        burst_seq.start(m_sequencer);
      end

      // Interleave with single items
      `uvm_info("NESTED_SEQ", "Sending final single item", UVM_MEDIUM)
      single_item = constrained_item::type_id::create("single_item");
      start_item(single_item);
      if (!single_item.randomize() with { write == 1'b0; })  // Read
        `uvm_error("NESTED_SEQ", "Final item randomization failed")
      finish_item(single_item);

      `uvm_info("NESTED_SEQ", "Nested sequence complete", UVM_MEDIUM)
    endtask
  endclass

  // Parallel sub-sequences using fork/join
  class parallel_sequence extends uvm_sequence #(constrained_item);
    `uvm_object_utils(parallel_sequence)

    function new(string name = "parallel_sequence");
      super.new(name);
    endfunction

    virtual task body();
      burst_sequence burst_a, burst_b;

      `uvm_info("PARALLEL_SEQ", "Starting parallel sub-sequences", UVM_MEDIUM)

      // Create sub-sequences with different configurations
      burst_a = burst_sequence::type_id::create("burst_a");
      burst_b = burst_sequence::type_id::create("burst_b");

      burst_a.base_addr = 16'h2000;
      burst_b.base_addr = 16'h3000;

      void'(burst_a.randomize() with { burst_count == 4; });
      void'(burst_b.randomize() with { burst_count == 4; });

      // Run sub-sequences in parallel using fork/join
      fork
        begin : BURST_A
          `uvm_info("PARALLEL_SEQ", "Starting burst_a", UVM_HIGH)
          burst_a.start(m_sequencer);
        end
        begin : BURST_B
          `uvm_info("PARALLEL_SEQ", "Starting burst_b", UVM_HIGH)
          burst_b.start(m_sequencer);
        end
      join

      `uvm_info("PARALLEL_SEQ", "Parallel sub-sequences complete", UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // SECTION 4: start_item/finish_item pattern variations
  //==========================================================================

  class item_pattern_sequence extends uvm_sequence #(constrained_item);
    `uvm_object_utils(item_pattern_sequence)

    function new(string name = "item_pattern_sequence");
      super.new(name);
    endfunction

    virtual task body();
      constrained_item item;

      `uvm_info("ITEM_PAT", "Demonstrating start_item/finish_item patterns", UVM_MEDIUM)

      // Pattern 1: Standard pattern - randomize between start_item and finish_item
      `uvm_info("ITEM_PAT", "Pattern 1: Standard randomize between start/finish", UVM_HIGH)
      item = constrained_item::type_id::create("item1");
      start_item(item);
      void'(item.randomize());
      finish_item(item);

      // Pattern 2: Pre-randomize before start_item
      `uvm_info("ITEM_PAT", "Pattern 2: Pre-randomize before start_item", UVM_HIGH)
      item = constrained_item::type_id::create("item2");
      void'(item.randomize());  // Randomize first
      start_item(item);         // Then request sequencer access
      finish_item(item);        // Send the item

      // Pattern 3: Inline constraint override
      `uvm_info("ITEM_PAT", "Pattern 3: Inline constraint override", UVM_HIGH)
      item = constrained_item::type_id::create("item3");
      start_item(item);
      void'(item.randomize() with {
        data == 8'hAB;          // Force specific data
        addr == 16'h4000;       // Force specific address
        write == 1'b1;
      });
      finish_item(item);

      // Pattern 4: Multiple items in rapid succession
      `uvm_info("ITEM_PAT", "Pattern 4: Back-to-back items", UVM_HIGH)
      for (int i = 0; i < 3; i++) begin
        item = constrained_item::type_id::create($sformatf("item4_%0d", i));
        start_item(item);
        void'(item.randomize() with { addr == 16'h5000 + i * 4; });
        finish_item(item);
      end

      // Pattern 5: Reusing item (not recommended, but valid)
      `uvm_info("ITEM_PAT", "Pattern 5: Item reuse with re-randomize", UVM_HIGH)
      item = constrained_item::type_id::create("reused_item");
      for (int i = 0; i < 2; i++) begin
        start_item(item);
        void'(item.randomize() with { addr == 16'h6000 + i * 4; });
        finish_item(item);
      end

      `uvm_info("ITEM_PAT", "Item pattern demonstrations complete", UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // SECTION 5: get_response pattern
  //==========================================================================

  // Driver that sends responses back
  class responding_driver extends uvm_driver #(constrained_item, response_item);
    `uvm_component_utils(responding_driver)

    int items_processed = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      constrained_item req_item;
      response_item rsp_item;

      forever begin
        // Get request from sequencer
        seq_item_port.get_next_item(req_item);

        `uvm_info("RSP_DRV", $sformatf("Processing: %s", req_item.convert2string()), UVM_HIGH)

        // Simulate processing time
        #10ns;

        // Create and send response
        rsp_item = response_item::type_id::create("rsp");
        rsp_item.req_transaction_id = req_item.transaction_id;

        // Simulate response data
        if (req_item.write) begin
          rsp_item.data = 8'h00;  // Write acknowledgment
          rsp_item.status = 2'b00;  // OK
        end else begin
          rsp_item.data = req_item.data ^ 8'hFF;  // Return inverted data for reads
          rsp_item.status = 2'b00;  // OK
        end

        // Copy sequence info for proper routing
        rsp_item.set_id_info(req_item);

        // Send response back to sequence
        seq_item_port.put_response(rsp_item);

        items_processed++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  // Sequence that uses get_response
  class response_sequence extends uvm_sequence #(constrained_item, response_item);
    `uvm_object_utils(response_sequence)

    int num_transactions = 5;
    int responses_received = 0;
    int responses_matched = 0;

    function new(string name = "response_sequence");
      super.new(name);
    endfunction

    virtual task body();
      constrained_item req;
      response_item rsp;

      `uvm_info("RSP_SEQ", $sformatf("Starting response sequence with %0d transactions",
                                     num_transactions), UVM_MEDIUM)

      // Enable response handling
      use_response_handler(1);

      for (int i = 0; i < num_transactions; i++) begin
        req = constrained_item::type_id::create($sformatf("req_%0d", i));

        start_item(req);
        if (!req.randomize())
          `uvm_error("RSP_SEQ", "Randomization failed")
        req.transaction_id = i;
        finish_item(req);

        `uvm_info("RSP_SEQ", $sformatf("Sent request: %s", req.convert2string()), UVM_HIGH)

        // Get response (blocking call)
        get_response(rsp);
        responses_received++;

        `uvm_info("RSP_SEQ", $sformatf("Got response: %s", rsp.convert2string()), UVM_HIGH)

        // Verify response matches request
        if (rsp.req_transaction_id == req.transaction_id) begin
          responses_matched++;
          `uvm_info("RSP_SEQ", $sformatf("Response matched for transaction %0d", i), UVM_HIGH)
        end else begin
          `uvm_error("RSP_SEQ", $sformatf("Response mismatch: expected id=%0d, got id=%0d",
                                          req.transaction_id, rsp.req_transaction_id))
        end
      end

      `uvm_info("RSP_SEQ", $sformatf("Response sequence complete: %0d/%0d responses matched",
                                     responses_matched, responses_received), UVM_MEDIUM)
    endtask
  endclass

  // Sequence that uses non-blocking response checking
  class async_response_sequence extends uvm_sequence #(constrained_item, response_item);
    `uvm_object_utils(async_response_sequence)

    function new(string name = "async_response_sequence");
      super.new(name);
    endfunction

    virtual task body();
      constrained_item req;
      response_item rsp;
      int outstanding_count = 0;

      `uvm_info("ASYNC_RSP", "Starting async response sequence", UVM_MEDIUM)

      // Send multiple requests, then collect responses
      fork
        begin : SEND_REQUESTS
          for (int i = 0; i < 4; i++) begin
            req = constrained_item::type_id::create($sformatf("req_%0d", i));
            start_item(req);
            void'(req.randomize());
            req.transaction_id = i;
            finish_item(req);
            outstanding_count++;
            `uvm_info("ASYNC_RSP", $sformatf("Sent request %0d", i), UVM_HIGH)
          end
        end

        begin : COLLECT_RESPONSES
          // Small delay to let requests go out first
          #5ns;
          while (outstanding_count > 0 || responses_waiting()) begin
            if (responses_waiting()) begin
              get_response(rsp);
              outstanding_count--;
              `uvm_info("ASYNC_RSP", $sformatf("Got response: %s", rsp.convert2string()), UVM_HIGH)
            end else begin
              #1ns;
            end
          end
        end
      join

      `uvm_info("ASYNC_RSP", "Async response sequence complete", UVM_MEDIUM)
    endtask

    // Helper to check if responses are waiting
    function bit responses_waiting();
      return response_queue.size() > 0;
    endfunction
  endclass

  //==========================================================================
  // SECTION 6: Virtual sequences
  //==========================================================================

  // Separate transaction type for a second interface
  class secondary_item extends uvm_sequence_item;
    `uvm_object_utils(secondary_item)

    rand bit [31:0] payload;
    rand bit [7:0] tag;

    constraint c_tag { tag < 8'h10; }

    function new(string name = "secondary_item");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("SEC tag=0x%02h payload=0x%08h", tag, payload);
    endfunction
  endclass

  // Secondary driver
  class secondary_driver extends uvm_driver #(secondary_item);
    `uvm_component_utils(secondary_driver)

    int items_processed = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        seq_item_port.get_next_item(req);
        `uvm_info("SEC_DRV", $sformatf("Processing: %s", req.convert2string()), UVM_HIGH)
        #8ns;
        items_processed++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  // Sequence for secondary interface
  class secondary_sequence extends uvm_sequence #(secondary_item);
    `uvm_object_utils(secondary_sequence)

    int num_items = 3;

    function new(string name = "secondary_sequence");
      super.new(name);
    endfunction

    virtual task body();
      secondary_item item;

      `uvm_info("SEC_SEQ", $sformatf("Starting secondary sequence (%0d items)", num_items), UVM_MEDIUM)

      repeat (num_items) begin
        item = secondary_item::type_id::create("item");
        start_item(item);
        void'(item.randomize());
        finish_item(item);
      end

      `uvm_info("SEC_SEQ", "Secondary sequence complete", UVM_MEDIUM)
    endtask
  endclass

  // Virtual sequencer - coordinates multiple sequencers
  class my_virtual_sequencer extends uvm_sequencer #(uvm_sequence_item);
    `uvm_component_utils(my_virtual_sequencer)

    // Handles to sub-sequencers
    uvm_sequencer #(constrained_item) primary_seqr;
    uvm_sequencer #(secondary_item) secondary_seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  // Base virtual sequence with p_sequencer access
  class virtual_base_sequence extends uvm_sequence #(uvm_sequence_item);
    `uvm_object_utils(virtual_base_sequence)
    `uvm_declare_p_sequencer(my_virtual_sequencer)

    function new(string name = "virtual_base_sequence");
      super.new(name);
    endfunction

    virtual task body();
      // Validate p_sequencer cast
      if (p_sequencer == null) begin
        `uvm_error("VSEQ", "p_sequencer is null - cast failed")
      end else begin
        `uvm_info("VSEQ", "Virtual sequencer access validated", UVM_HIGH)
      end
    endtask
  endclass

  // Coordinated virtual sequence - runs sequences on multiple interfaces
  class coordinated_virtual_sequence extends virtual_base_sequence;
    `uvm_object_utils(coordinated_virtual_sequence)

    function new(string name = "coordinated_virtual_sequence");
      super.new(name);
    endfunction

    virtual task body();
      basic_sequence primary_seq;
      secondary_sequence secondary_seq;

      // Call base class body to validate p_sequencer
      super.body();

      `uvm_info("COORD_VSEQ", "Starting coordinated virtual sequence", UVM_MEDIUM)

      // Pattern 1: Sequential - primary then secondary
      `uvm_info("COORD_VSEQ", "Phase 1: Sequential execution", UVM_HIGH)
      primary_seq = basic_sequence::type_id::create("primary_seq");
      primary_seq.num_items = 2;
      primary_seq.start(p_sequencer.primary_seqr);

      secondary_seq = secondary_sequence::type_id::create("secondary_seq");
      secondary_seq.num_items = 2;
      secondary_seq.start(p_sequencer.secondary_seqr);

      // Pattern 2: Parallel - both interfaces at once
      `uvm_info("COORD_VSEQ", "Phase 2: Parallel execution", UVM_HIGH)
      fork
        begin
          basic_sequence par_primary;
          par_primary = basic_sequence::type_id::create("par_primary");
          par_primary.num_items = 3;
          par_primary.start(p_sequencer.primary_seqr);
        end
        begin
          secondary_sequence par_secondary;
          par_secondary = secondary_sequence::type_id::create("par_secondary");
          par_secondary.num_items = 3;
          par_secondary.start(p_sequencer.secondary_seqr);
        end
      join

      `uvm_info("COORD_VSEQ", "Coordinated virtual sequence complete", UVM_MEDIUM)
    endtask
  endclass

  // Virtual sequence with slave responders (fork/join_none pattern)
  class master_slave_virtual_sequence extends virtual_base_sequence;
    `uvm_object_utils(master_slave_virtual_sequence)

    int num_master_transactions = 5;

    function new(string name = "master_slave_virtual_sequence");
      super.new(name);
    endfunction

    virtual task body();
      basic_sequence master_seq;

      super.body();

      `uvm_info("MS_VSEQ", "Starting master/slave virtual sequence", UVM_MEDIUM)

      // Start slave responder in background (fork/join_none pattern)
      fork
        begin : SLAVE_RESPONDER
          secondary_sequence slave_resp;
          forever begin
            slave_resp = secondary_sequence::type_id::create("slave_resp");
            slave_resp.num_items = 1;
            slave_resp.start(p_sequencer.secondary_seqr);
          end
        end
      join_none

      // Run master sequence (foreground)
      master_seq = basic_sequence::type_id::create("master_seq");
      master_seq.num_items = num_master_transactions;
      master_seq.start(p_sequencer.primary_seqr);

      // Disable the slave responder after master completes
      disable fork;

      `uvm_info("MS_VSEQ", "Master/slave virtual sequence complete", UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // Test Infrastructure
  //==========================================================================

  // Primary driver
  class primary_driver extends uvm_driver #(constrained_item);
    `uvm_component_utils(primary_driver)

    int items_processed = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        seq_item_port.get_next_item(req);
        `uvm_info("PRI_DRV", $sformatf("Processing: %s", req.convert2string()), UVM_HIGH)
        #10ns;
        items_processed++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  // Test environment
  class test_env extends uvm_env;
    `uvm_component_utils(test_env)

    // Primary interface components
    primary_driver pri_drv;
    uvm_sequencer #(constrained_item) pri_seqr;

    // Secondary interface components
    secondary_driver sec_drv;
    uvm_sequencer #(secondary_item) sec_seqr;

    // Response-capable components
    responding_driver rsp_drv;
    uvm_sequencer #(constrained_item, response_item) rsp_seqr;

    // Virtual sequencer
    my_virtual_sequencer v_seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Create primary interface
      pri_drv = primary_driver::type_id::create("pri_drv", this);
      pri_seqr = uvm_sequencer #(constrained_item)::type_id::create("pri_seqr", this);

      // Create secondary interface
      sec_drv = secondary_driver::type_id::create("sec_drv", this);
      sec_seqr = uvm_sequencer #(secondary_item)::type_id::create("sec_seqr", this);

      // Create response-capable interface
      rsp_drv = responding_driver::type_id::create("rsp_drv", this);
      rsp_seqr = uvm_sequencer #(constrained_item, response_item)::type_id::create("rsp_seqr", this);

      // Create virtual sequencer
      v_seqr = my_virtual_sequencer::type_id::create("v_seqr", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);

      // Connect primary interface
      pri_drv.seq_item_port.connect(pri_seqr.seq_item_export);

      // Connect secondary interface
      sec_drv.seq_item_port.connect(sec_seqr.seq_item_export);

      // Connect response interface
      rsp_drv.seq_item_port.connect(rsp_seqr.seq_item_export);

      // Connect virtual sequencer to sub-sequencers
      v_seqr.primary_seqr = pri_seqr;
      v_seqr.secondary_seqr = sec_seqr;
    endfunction
  endclass

  //==========================================================================
  // Individual Feature Tests
  //==========================================================================

  // Test 1: Constrained items
  class constraint_test extends uvm_test;
    `uvm_component_utils(constraint_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      basic_sequence seq;

      phase.raise_objection(this, "Running constraint test");

      `uvm_info("TEST", "=== Test 1: Constrained Items ===", UVM_NONE)

      seq = basic_sequence::type_id::create("seq");
      seq.num_items = 10;
      seq.start(env.pri_seqr);

      `uvm_info("TEST", $sformatf("Driver processed %0d items", env.pri_drv.items_processed), UVM_NONE)

      phase.drop_objection(this, "Constraint test complete");
    endtask
  endclass

  // Test 2: Nested sequences
  class nested_test extends uvm_test;
    `uvm_component_utils(nested_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      nested_sequence nested_seq;
      parallel_sequence parallel_seq;

      phase.raise_objection(this, "Running nested test");

      `uvm_info("TEST", "=== Test 2: Nested Sequences ===", UVM_NONE)

      // Test nested sequences
      `uvm_info("TEST", "--- Testing nested sequence ---", UVM_MEDIUM)
      nested_seq = nested_sequence::type_id::create("nested_seq");
      nested_seq.num_bursts = 2;
      nested_seq.start(env.pri_seqr);

      // Test parallel sub-sequences
      `uvm_info("TEST", "--- Testing parallel sub-sequences ---", UVM_MEDIUM)
      parallel_seq = parallel_sequence::type_id::create("parallel_seq");
      parallel_seq.start(env.pri_seqr);

      `uvm_info("TEST", $sformatf("Driver processed %0d items", env.pri_drv.items_processed), UVM_NONE)

      phase.drop_objection(this, "Nested test complete");
    endtask
  endclass

  // Test 3: Item patterns
  class item_pattern_test extends uvm_test;
    `uvm_component_utils(item_pattern_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      item_pattern_sequence seq;

      phase.raise_objection(this, "Running item pattern test");

      `uvm_info("TEST", "=== Test 3: Item Patterns ===", UVM_NONE)

      seq = item_pattern_sequence::type_id::create("seq");
      seq.start(env.pri_seqr);

      `uvm_info("TEST", $sformatf("Driver processed %0d items", env.pri_drv.items_processed), UVM_NONE)

      phase.drop_objection(this, "Item pattern test complete");
    endtask
  endclass

  // Test 4: Response handling
  class response_test extends uvm_test;
    `uvm_component_utils(response_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      response_sequence rsp_seq;

      phase.raise_objection(this, "Running response test");

      `uvm_info("TEST", "=== Test 4: Response Handling ===", UVM_NONE)

      rsp_seq = response_sequence::type_id::create("rsp_seq");
      rsp_seq.num_transactions = 5;
      rsp_seq.start(env.rsp_seqr);

      `uvm_info("TEST", $sformatf("Responses received: %0d, matched: %0d",
                                  rsp_seq.responses_received, rsp_seq.responses_matched), UVM_NONE)

      if (rsp_seq.responses_matched == rsp_seq.responses_received)
        `uvm_info("TEST", "Response test PASSED", UVM_NONE)
      else
        `uvm_error("TEST", "Response test FAILED - mismatched responses")

      phase.drop_objection(this, "Response test complete");
    endtask
  endclass

  // Test 5: Virtual sequences
  class virtual_seq_test extends uvm_test;
    `uvm_component_utils(virtual_seq_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      coordinated_virtual_sequence coord_seq;
      master_slave_virtual_sequence ms_seq;

      phase.raise_objection(this, "Running virtual sequence test");

      `uvm_info("TEST", "=== Test 5: Virtual Sequences ===", UVM_NONE)

      // Test coordinated virtual sequence
      `uvm_info("TEST", "--- Testing coordinated virtual sequence ---", UVM_MEDIUM)
      coord_seq = coordinated_virtual_sequence::type_id::create("coord_seq");
      coord_seq.start(env.v_seqr);

      // Test master/slave virtual sequence
      `uvm_info("TEST", "--- Testing master/slave virtual sequence ---", UVM_MEDIUM)
      ms_seq = master_slave_virtual_sequence::type_id::create("ms_seq");
      ms_seq.num_master_transactions = 3;
      ms_seq.start(env.v_seqr);

      `uvm_info("TEST", $sformatf("Primary driver: %0d items", env.pri_drv.items_processed), UVM_NONE)
      `uvm_info("TEST", $sformatf("Secondary driver: %0d items", env.sec_drv.items_processed), UVM_NONE)

      phase.drop_objection(this, "Virtual sequence test complete");
    endtask
  endclass

  //==========================================================================
  // Comprehensive Test - runs all patterns
  //==========================================================================

  class comprehensive_sequence_test extends uvm_test;
    `uvm_component_utils(comprehensive_sequence_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      basic_sequence basic_seq;
      nested_sequence nested_seq;
      item_pattern_sequence pattern_seq;
      response_sequence rsp_seq;
      coordinated_virtual_sequence vseq;

      phase.raise_objection(this, "Running comprehensive sequence test");

      `uvm_info("TEST", "========================================", UVM_NONE)
      `uvm_info("TEST", "UVM Sequence Patterns Comprehensive Test", UVM_NONE)
      `uvm_info("TEST", "========================================", UVM_NONE)

      // Section 1 & 2: Basic sequence with constrained items
      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Section 1&2: Constrained Items & body() Task ===", UVM_NONE)
      basic_seq = basic_sequence::type_id::create("basic_seq");
      basic_seq.num_items = 5;
      basic_seq.start(env.pri_seqr);
      `uvm_info("TEST", $sformatf("Completed: %0d items processed", env.pri_drv.items_processed), UVM_NONE)

      // Section 3: Nested sequences
      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Section 3: Nested Sequences ===", UVM_NONE)
      nested_seq = nested_sequence::type_id::create("nested_seq");
      nested_seq.num_bursts = 2;
      nested_seq.start(env.pri_seqr);
      `uvm_info("TEST", $sformatf("Completed: %0d total items processed", env.pri_drv.items_processed), UVM_NONE)

      // Section 4: Item patterns
      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Section 4: start_item/finish_item Patterns ===", UVM_NONE)
      pattern_seq = item_pattern_sequence::type_id::create("pattern_seq");
      pattern_seq.start(env.pri_seqr);
      `uvm_info("TEST", $sformatf("Completed: %0d total items processed", env.pri_drv.items_processed), UVM_NONE)

      // Section 5: Response handling
      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Section 5: get_response Pattern ===", UVM_NONE)
      rsp_seq = response_sequence::type_id::create("rsp_seq");
      rsp_seq.num_transactions = 5;
      rsp_seq.start(env.rsp_seqr);
      `uvm_info("TEST", $sformatf("Responses: %0d received, %0d matched",
                                  rsp_seq.responses_received, rsp_seq.responses_matched), UVM_NONE)

      // Section 6: Virtual sequences
      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Section 6: Virtual Sequences ===", UVM_NONE)
      vseq = coordinated_virtual_sequence::type_id::create("vseq");
      vseq.start(env.v_seqr);
      `uvm_info("TEST", $sformatf("Primary: %0d, Secondary: %0d items",
                                  env.pri_drv.items_processed, env.sec_drv.items_processed), UVM_NONE)

      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "========================================", UVM_NONE)
      `uvm_info("TEST", "Comprehensive Test Complete", UVM_NONE)
      `uvm_info("TEST", "========================================", UVM_NONE)

      phase.drop_objection(this, "Comprehensive sequence test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);

      `uvm_info("TEST", "", UVM_NONE)
      `uvm_info("TEST", "=== Final Statistics ===", UVM_NONE)
      `uvm_info("TEST", $sformatf("Primary driver items: %0d", env.pri_drv.items_processed), UVM_NONE)
      `uvm_info("TEST", $sformatf("Secondary driver items: %0d", env.sec_drv.items_processed), UVM_NONE)
      `uvm_info("TEST", $sformatf("Response driver items: %0d", env.rsp_drv.items_processed), UVM_NONE)

      if (env.pri_drv.items_processed > 0 &&
          env.sec_drv.items_processed > 0 &&
          env.rsp_drv.items_processed > 0) begin
        `uvm_info("TEST", "ALL SEQUENCE PATTERN TESTS PASSED", UVM_NONE)
      end else begin
        `uvm_error("TEST", "TEST FAILED - some drivers received no items")
      end
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import sequence_test_pkg::*;

  initial begin
    `uvm_info("TB", "UVM Sequence Patterns Test", UVM_NONE)
    run_test("comprehensive_sequence_test");
  end

endmodule
