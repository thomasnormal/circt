//===----------------------------------------------------------------------===//
// UVM Stress Test - Comprehensive edge case testing for UVM stub
//===----------------------------------------------------------------------===//
// This test exercises multiple UVM features simultaneously to find edge cases:
// 1. Complex config_db patterns (wildcards, hierarchical paths, nested contexts)
// 2. Sequence library patterns (sequence_item, sequences, virtual sequences)
// 3. Coverage collection patterns (covergroups, functional coverage)
// 4. Report server patterns (custom servers, catchers, message filtering)
// 5. Phase jumping patterns (phase state, wait_for_state, objections)
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package uvm_stress_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // PART 1: Complex config_db patterns
  //==========================================================================

  // Nested configuration object
  class inner_config extends uvm_object;
    `uvm_object_utils(inner_config)

    int depth;
    string tag;
    bit enabled;

    function new(string name = "inner_config");
      super.new(name);
      depth = 0;
      tag = "";
      enabled = 1;
    endfunction

    virtual function string convert2string();
      return $sformatf("depth=%0d tag=%s enabled=%0b", depth, tag, enabled);
    endfunction
  endclass

  class outer_config extends uvm_object;
    `uvm_object_utils(outer_config)

    inner_config inner_cfg;
    int array_data[10];
    string name_list[$];

    function new(string name = "outer_config");
      super.new(name);
      inner_cfg = new("inner_cfg");
      for (int i = 0; i < 10; i++) array_data[i] = i;
    endfunction

    virtual function string convert2string();
      return $sformatf("inner=[%s] array_data[0]=%0d names=%0d",
                       inner_cfg.convert2string(), array_data[0], name_list.size());
    endfunction
  endclass

  // Config_db stress tester
  class config_db_stress extends uvm_component;
    `uvm_component_utils(config_db_stress)

    int test_pass = 0;
    int test_fail = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void check(bit cond, string name);
      if (cond) begin
        test_pass++;
        `uvm_info("PASS", name, UVM_HIGH)
      end else begin
        test_fail++;
        `uvm_error("FAIL", name)
      end
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      run_config_db_stress_tests();
    endfunction

    function void run_config_db_stress_tests();
      int val;
      string str;
      outer_config ocfg;
      inner_config icfg;

      `uvm_info("STRESS", "=== Config DB Stress Tests ===", UVM_LOW)

      // Test 1: Deeply nested wildcards
      uvm_config_db#(int)::set(null, "*.*.*.deep", "field1", 111);
      uvm_config_db#(int)::set(null, "a.b.c.deep", "field1", 222);
      check(uvm_config_db#(int)::get(null, "a.b.c.deep", "field1", val) && val == 222,
            "Exact match over nested wildcards");
      check(uvm_config_db#(int)::get(null, "x.y.z.deep", "field1", val) && val == 111,
            "Nested wildcards match other paths");

      // Test 2: Multiple overlapping wildcards
      uvm_config_db#(int)::set(null, "*", "overlap", 1);
      uvm_config_db#(int)::set(null, "a.*", "overlap", 2);
      uvm_config_db#(int)::set(null, "a.b.*", "overlap", 3);
      uvm_config_db#(int)::set(null, "a.b.c", "overlap", 4);
      check(uvm_config_db#(int)::get(null, "a.b.c", "overlap", val) && val == 4,
            "Most specific wins (exact)");
      check(uvm_config_db#(int)::get(null, "a.b.x", "overlap", val) && val == 3,
            "More specific wildcard wins (a.b.*)");
      check(uvm_config_db#(int)::get(null, "a.x", "overlap", val) && val == 2,
            "Middle wildcard wins (a.*)");
      check(uvm_config_db#(int)::get(null, "x", "overlap", val) && val == 1,
            "Global wildcard for unmatched");

      // Test 3: Question mark wildcards
      uvm_config_db#(int)::set(null, "agent?", "qmark", 100);
      uvm_config_db#(int)::set(null, "agent??", "qmark", 200);
      check(uvm_config_db#(int)::get(null, "agent1", "qmark", val) && val == 100,
            "Single ? matches single char");
      check(uvm_config_db#(int)::get(null, "agent12", "qmark", val) && val == 200,
            "Double ?? matches two chars");

      // Test 4: Complex object configuration
      ocfg = new("stress_outer");
      ocfg.inner_cfg.depth = 5;
      ocfg.inner_cfg.tag = "stress_test";
      ocfg.name_list.push_back("item1");
      ocfg.name_list.push_back("item2");
      uvm_config_db#(outer_config)::set(null, "*", "complex_cfg", ocfg);
      begin
        outer_config retrieved;
        check(uvm_config_db#(outer_config)::get(null, "any.path", "complex_cfg", retrieved) &&
              retrieved.inner_cfg.depth == 5 &&
              retrieved.inner_cfg.tag == "stress_test" &&
              retrieved.name_list.size() == 2,
              "Complex nested object config");
      end

      // Test 5: Empty and special paths
      uvm_config_db#(string)::set(null, "", "empty_path", "value1");
      check(uvm_config_db#(string)::get(null, "", "empty_path", str) && str == "value1",
            "Empty path set/get");

      // Test 6: Overwrite in different scopes
      uvm_config_db#(int)::set(null, "scope.a", "shared", 10);
      uvm_config_db#(int)::set(null, "scope.b", "shared", 20);
      check(uvm_config_db#(int)::get(null, "scope.a", "shared", val) && val == 10,
            "Scope isolation - scope.a");
      check(uvm_config_db#(int)::get(null, "scope.b", "shared", val) && val == 20,
            "Scope isolation - scope.b");

      // Test 7: Regex-like patterns with uvm_is_match
      check(uvm_is_match("uvm_test_top.env.agent[*]", "uvm_test_top.env.agent[0]"),
            "Array-like path matching");
      check(uvm_is_match("*monitor*", "env.my_monitor.analysis"),
            "Contains pattern in middle of path");

      `uvm_info("STRESS", $sformatf("Config DB: %0d passed, %0d failed",
                                    test_pass, test_fail), UVM_LOW)
    endfunction
  endclass

  //==========================================================================
  // PART 2: Sequence Library Patterns
  //==========================================================================

  // Complex sequence item with constraints
  class stress_item extends uvm_sequence_item;
    `uvm_object_utils(stress_item)

    rand bit [31:0] addr;
    rand bit [63:0] data;
    rand bit [3:0]  burst_len;
    rand bit        write;
    rand bit [7:0]  id;

    constraint c_addr_align { addr[1:0] == 2'b00; }
    constraint c_burst { burst_len inside {[1:15]}; }
    constraint c_id_range { id < 16; }

    function new(string name = "stress_item");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("id=%0h addr=%08h data=%016h burst=%0d write=%0b",
                       id, addr, data, burst_len, write);
    endfunction
  endclass

  // Response item
  class stress_rsp extends uvm_sequence_item;
    `uvm_object_utils(stress_rsp)

    bit [7:0]  id;
    bit [63:0] data;
    bit        error;

    function new(string name = "stress_rsp");
      super.new(name);
    endfunction
  endclass

  // Base sequence with response handling
  class base_stress_seq extends uvm_sequence #(stress_item, stress_rsp);
    `uvm_object_utils(base_stress_seq)

    int items_sent = 0;
    int responses_received = 0;

    function new(string name = "base_stress_seq");
      super.new(name);
    endfunction

    virtual task body();
      `uvm_error("SEQ", "base_stress_seq::body() should be overridden")
    endtask
  endclass

  // Sequence with back-to-back items
  class burst_sequence extends base_stress_seq;
    `uvm_object_utils(burst_sequence)

    rand int burst_count;
    constraint c_burst { burst_count inside {[4:16]}; }

    function new(string name = "burst_sequence");
      super.new(name);
    endfunction

    virtual task body();
      stress_item item;
      `uvm_info("SEQ", $sformatf("Starting burst sequence with %0d items", burst_count), UVM_MEDIUM)

      repeat (burst_count) begin
        item = stress_item::type_id::create("item");
        start_item(item);
        if (!item.randomize()) `uvm_error("SEQ", "Randomization failed")
        finish_item(item);
        items_sent++;
      end
    endtask
  endclass

  // Sequence with interleaved lock/unlock
  class lock_stress_sequence extends base_stress_seq;
    `uvm_object_utils(lock_stress_sequence)

    function new(string name = "lock_stress_sequence");
      super.new(name);
    endfunction

    virtual task body();
      stress_item item;

      // First phase: unlocked
      `uvm_info("SEQ", "Phase 1: Unlocked items", UVM_MEDIUM)
      repeat (2) begin
        item = stress_item::type_id::create("item");
        start_item(item);
        void'(item.randomize());
        finish_item(item);
        items_sent++;
      end

      // Second phase: locked
      `uvm_info("SEQ", "Phase 2: Locked items", UVM_MEDIUM)
      lock();
      if (!has_lock()) `uvm_error("SEQ", "Failed to acquire lock")
      repeat (3) begin
        item = stress_item::type_id::create("item");
        start_item(item);
        void'(item.randomize());
        finish_item(item);
        items_sent++;
      end
      unlock();

      // Third phase: grab and release
      `uvm_info("SEQ", "Phase 3: Grabbed items", UVM_MEDIUM)
      grab();
      repeat (2) begin
        item = stress_item::type_id::create("item");
        start_item(item);
        void'(item.randomize());
        finish_item(item);
        items_sent++;
      end
      ungrab();

      `uvm_info("SEQ", $sformatf("Sequence complete: %0d items sent", items_sent), UVM_MEDIUM)
    endtask
  endclass

  // Virtual sequence with sub-sequencers
  class virtual_stress_sequence extends uvm_sequence #(uvm_sequence_item);
    `uvm_object_utils(virtual_stress_sequence)

    // Virtual sequencer handle (in real usage, set via p_sequencer)
    uvm_sequencer #(stress_item, stress_rsp) sub_seqr1;
    uvm_sequencer #(stress_item, stress_rsp) sub_seqr2;

    function new(string name = "virtual_stress_sequence");
      super.new(name);
    endfunction

    virtual task body();
      burst_sequence seq1;
      lock_stress_sequence seq2;

      if (sub_seqr1 == null || sub_seqr2 == null) begin
        `uvm_warning("VSEQ", "Sub-sequencers not set, skipping parallel execution")
        return;
      end

      `uvm_info("VSEQ", "Starting virtual sequence with parallel sub-sequences", UVM_MEDIUM)

      fork
        begin
          seq1 = burst_sequence::type_id::create("seq1");
          seq1.burst_count = 5;
          seq1.start(sub_seqr1);
        end
        begin
          seq2 = lock_stress_sequence::type_id::create("seq2");
          seq2.start(sub_seqr2);
        end
      join

      `uvm_info("VSEQ", "Virtual sequence complete", UVM_MEDIUM)
    endtask
  endclass

  // Nested sequence (sequence calling other sequences)
  class nested_sequence extends base_stress_seq;
    `uvm_object_utils(nested_sequence)

    int nesting_depth;

    function new(string name = "nested_sequence");
      super.new(name);
      nesting_depth = 0;
    endfunction

    virtual task body();
      stress_item item;
      nested_sequence child_seq;

      `uvm_info("SEQ", $sformatf("Nested sequence at depth %0d", nesting_depth), UVM_MEDIUM)

      // Send some items at this level
      repeat (2) begin
        item = stress_item::type_id::create("item");
        start_item(item);
        void'(item.randomize());
        finish_item(item);
        items_sent++;
      end

      // Recurse if not too deep
      if (nesting_depth < 3) begin
        child_seq = nested_sequence::type_id::create("child");
        child_seq.nesting_depth = nesting_depth + 1;
        child_seq.start(m_sequencer, this);
        items_sent += child_seq.items_sent;
      end
    endtask
  endclass

  //==========================================================================
  // PART 3: Coverage Collection Patterns
  //==========================================================================

  class coverage_stress extends uvm_subscriber #(stress_item);
    `uvm_component_utils(coverage_stress)

    // Covergroup with complex bins
    bit [31:0] sampled_addr;
    bit [3:0]  sampled_burst;
    bit        sampled_write;
    int        sample_count = 0;

    covergroup stress_cg;
      option.per_instance = 1;

      addr_cp: coverpoint sampled_addr[31:28] {
        bins low = {[0:3]};
        bins mid = {[4:11]};
        bins high = {[12:15]};
        illegal_bins reserved = {14, 15};
      }

      burst_cp: coverpoint sampled_burst {
        bins single = {1};
        bins short_burst = {[2:4]};
        bins medium_burst = {[5:8]};
        bins long_burst = {[9:15]};
      }

      write_cp: coverpoint sampled_write {
        bins reads = {0};
        bins writes = {1};
      }

      // Cross coverage
      addr_write_cross: cross addr_cp, write_cp {
        ignore_bins ignore_high_write = binsof(addr_cp.high) && binsof(write_cp.writes);
      }

      burst_write_cross: cross burst_cp, write_cp;
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      stress_cg = new();
    endfunction

    virtual function void write(stress_item t);
      sample_count++;
      sampled_addr = t.addr;
      sampled_burst = t.burst_len;
      sampled_write = t.write;
      stress_cg.sample();

      if (sample_count % 10 == 0)
        `uvm_info("COV", $sformatf("Sampled %0d items, coverage=%.1f%%",
                                   sample_count, stress_cg.get_coverage()), UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("COV", $sformatf("Final coverage: %.2f%% (%0d samples)",
                                 stress_cg.get_coverage(), sample_count), UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // PART 4: Report Server Patterns
  //==========================================================================

  // Custom report catcher that counts and modifies messages
  class stress_catcher extends uvm_report_catcher;
    int caught_count = 0;
    int modified_count = 0;
    string filter_id;
    bit demote_errors = 0;

    function new(string name = "stress_catcher");
      super.new(name);
      filter_id = "";
    endfunction

    virtual function uvm_action_type_e catch_action();
      caught_count++;

      // Demote errors with specific ID to warnings
      if (demote_errors && get_id() == filter_id && get_severity() == UVM_ERROR) begin
        set_severity(UVM_WARNING);
        set_message({"[DEMOTED] ", get_message()});
        modified_count++;
        return CAUGHT;
      end

      // Count but don't modify
      return THROW;
    endfunction
  endclass

  // Component that exercises report server features
  class report_stress extends uvm_component;
    `uvm_component_utils(report_stress)

    int test_pass = 0;
    int test_fail = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void check(bit cond, string name);
      if (cond) begin test_pass++; `uvm_info("PASS", name, UVM_HIGH) end
      else begin test_fail++; `uvm_error("FAIL", name) end
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_report_server server;
      stress_catcher catcher;
      int initial_info, initial_warning, initial_error;

      phase.raise_objection(this, "Report stress test");

      `uvm_info("STRESS", "=== Report Server Stress Tests ===", UVM_LOW)

      server = uvm_report_server::get_server();
      initial_info = server.get_info_count();
      initial_warning = server.get_warning_count();
      initial_error = server.get_error_count();

      // Test 1: Add/remove catchers dynamically
      catcher = new("stress_catcher");
      catcher.filter_id = "DEMOTE_TEST";
      catcher.demote_errors = 1;
      uvm_report_catcher::add(catcher);
      check(uvm_report_catcher::get_catcher_count() > 0, "Catcher added");

      // Test 2: Max quit count manipulation
      server.set_max_quit_count(100);
      check(server.get_max_quit_count() == 100, "Max quit count set");

      // Test 3: Generate various severity messages
      `uvm_info("STRESS", "Info message 1", UVM_LOW)
      `uvm_info("STRESS", "Info message 2", UVM_MEDIUM)
      `uvm_info("STRESS", "Info message 3", UVM_HIGH)
      `uvm_warning("STRESS", "Warning message 1")

      // Test 4: Check counts increased
      check(server.get_info_count() > initial_info, "Info count increased");
      check(server.get_warning_count() > initial_warning, "Warning count increased");

      // Test 5: ID-based counting
      check(server.get_id_count("STRESS") > 0, "ID count tracking works");

      // Test 6: Catcher was invoked
      check(catcher.caught_count > 0, "Catcher caught messages");

      // Clean up catcher
      uvm_report_catcher::remove(catcher);

      // Test 7: Verbosity control
      set_report_verbosity_level(UVM_DEBUG);
      check(get_report_verbosity_level() == UVM_DEBUG, "Verbosity set to DEBUG");

      // Test 8: Action configuration
      set_report_severity_action(UVM_INFO, UVM_DISPLAY | UVM_LOG);
      set_report_id_action("SPECIAL", UVM_DISPLAY);

      `uvm_info("STRESS", $sformatf("Report Server: %0d passed, %0d failed",
                                    test_pass, test_fail), UVM_LOW)

      phase.drop_objection(this, "Report stress test done");
    endtask
  endclass

  //==========================================================================
  // PART 5: Phase Jumping Patterns
  //==========================================================================

  class phase_stress extends uvm_component;
    `uvm_component_utils(phase_stress)

    int phase_enter_count[string];
    uvm_phase saved_phase;
    int test_pass = 0;
    int test_fail = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void check(bit cond, string name);
      if (cond) begin test_pass++; `uvm_info("PASS", name, UVM_HIGH) end
      else begin test_fail++; `uvm_error("FAIL", name) end
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      phase_enter_count["build"]++;
      saved_phase = phase;
      `uvm_info("PHASE", "Entered build_phase", UVM_MEDIUM)
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      phase_enter_count["connect"]++;
      `uvm_info("PHASE", "Entered connect_phase", UVM_MEDIUM)
    endfunction

    virtual function void end_of_elaboration_phase(uvm_phase phase);
      super.end_of_elaboration_phase(phase);
      phase_enter_count["eoe"]++;
      `uvm_info("PHASE", "Entered end_of_elaboration_phase", UVM_MEDIUM)
    endfunction

    virtual function void start_of_simulation_phase(uvm_phase phase);
      super.start_of_simulation_phase(phase);
      phase_enter_count["sos"]++;
      `uvm_info("PHASE", "Entered start_of_simulation_phase", UVM_MEDIUM)
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_objection objection;
      uvm_phase_state state;
      int count;

      phase.raise_objection(this, "Phase stress test");
      phase_enter_count["run"]++;
      `uvm_info("PHASE", "Entered run_phase", UVM_MEDIUM)

      // Test phase objection methods
      objection = phase.get_objection();
      check(objection != null, "Got phase objection");

      count = phase.get_objection_count(this);
      check(count >= 1, "Objection count at least 1");

      // Test phase state
      state = phase.get_state();
      check(state == UVM_PHASE_EXECUTING || state == UVM_PHASE_STARTED,
            $sformatf("Phase state is EXECUTING or STARTED (got %s)", state.name()));

      // Test wait_for_state (stub just sets state immediately)
      phase.wait_for_state(UVM_PHASE_READY_TO_END);
      state = phase.get_state();
      check(state == UVM_PHASE_READY_TO_END, "wait_for_state changed state");

      // Test nested objections
      phase.raise_objection(this, "Nested objection 1");
      phase.raise_objection(this, "Nested objection 2");
      count = phase.get_objection_count(this);
      check(count >= 3, $sformatf("Multiple objections raised (count=%0d)", count));

      phase.drop_objection(this, "Drop nested 2");
      phase.drop_objection(this, "Drop nested 1");

      // Test global phase handles
      check(build_ph != null, "build_ph global handle exists");
      check(connect_ph != null, "connect_ph global handle exists");
      check(run_ph != null, "run_ph global handle exists");

      // Test uvm_test_done objection
      uvm_test_done.raise_objection(this, "Test completion objection");
      check(uvm_test_done.raised(), "uvm_test_done has raised objection");
      uvm_test_done.drop_objection(this, "Drop test completion");

      #10ns;
      `uvm_info("PHASE", $sformatf("Phase tests: %0d passed, %0d failed",
                                   test_pass, test_fail), UVM_LOW)

      phase.drop_objection(this, "Phase stress test done");
    endtask

    virtual function void extract_phase(uvm_phase phase);
      super.extract_phase(phase);
      phase_enter_count["extract"]++;
      `uvm_info("PHASE", "Entered extract_phase", UVM_MEDIUM)
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      phase_enter_count["check"]++;
      `uvm_info("PHASE", "Entered check_phase", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      phase_enter_count["report"]++;
      `uvm_info("PHASE", "Entered report_phase", UVM_MEDIUM)
      `uvm_info("PHASE", $sformatf("Phase entry counts: %p", phase_enter_count), UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // PART 6: Driver for stress sequences
  //==========================================================================

  class stress_driver extends uvm_driver #(stress_item, stress_rsp);
    `uvm_component_utils(stress_driver)

    int items_processed = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      stress_item item;
      stress_rsp response;

      forever begin
        seq_item_port.get_next_item(item);
        `uvm_info("DRV", $sformatf("Processing: %s", item.convert2string()), UVM_HIGH)

        // Simulate processing delay
        #1ns;
        items_processed++;

        // Create and send response
        response = stress_rsp::type_id::create("response");
        response.id = item.id;
        response.data = item.data;
        response.error = 0;

        seq_item_port.item_done(response);
      end
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("DRV", $sformatf("Processed %0d items", items_processed), UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // PART 7: Main Test Environment
  //==========================================================================

  class stress_agent extends uvm_agent;
    `uvm_component_utils(stress_agent)

    stress_driver driver;
    uvm_sequencer #(stress_item, stress_rsp) sequencer;
    uvm_analysis_port #(stress_item) ap;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (is_active == UVM_ACTIVE) begin
        driver = stress_driver::type_id::create("driver", this);
        sequencer = uvm_sequencer #(stress_item, stress_rsp)::type_id::create("sequencer", this);
      end
      ap = new("ap", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      if (is_active == UVM_ACTIVE) begin
        driver.seq_item_port.connect(sequencer.seq_item_export);
      end
    endfunction
  endclass

  class stress_env extends uvm_env;
    `uvm_component_utils(stress_env)

    stress_agent agent1;
    stress_agent agent2;
    coverage_stress cov;
    config_db_stress cfg_stress;
    report_stress rpt_stress;
    phase_stress ph_stress;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Configure agents via config_db
      uvm_config_db#(uvm_active_passive_enum)::set(this, "agent1", "is_active", UVM_ACTIVE);
      uvm_config_db#(uvm_active_passive_enum)::set(this, "agent2", "is_active", UVM_ACTIVE);

      agent1 = stress_agent::type_id::create("agent1", this);
      agent2 = stress_agent::type_id::create("agent2", this);
      cov = coverage_stress::type_id::create("cov", this);
      cfg_stress = config_db_stress::type_id::create("cfg_stress", this);
      rpt_stress = report_stress::type_id::create("rpt_stress", this);
      ph_stress = phase_stress::type_id::create("ph_stress", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      // Connect analysis ports to coverage
      agent1.ap.connect(cov.analysis_export);
      agent2.ap.connect(cov.analysis_export);
    endfunction
  endclass

  //==========================================================================
  // PART 8: Main Test
  //==========================================================================

  class uvm_stress_test extends uvm_test;
    `uvm_component_utils(uvm_stress_test)

    stress_env env;
    int total_pass = 0;
    int total_fail = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Set up complex config_db entries for the hierarchy
      uvm_config_db#(int)::set(this, "*", "global_timeout", 10000);
      uvm_config_db#(int)::set(this, "env.*", "env_setting", 42);
      uvm_config_db#(int)::set(this, "env.agent?", "agent_setting", 100);
      uvm_config_db#(string)::set(this, "env.*.driver", "driver_mode", "fast");

      env = stress_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      burst_sequence burst_seq;
      lock_stress_sequence lock_seq;
      nested_sequence nested_seq;
      virtual_stress_sequence virt_seq;

      phase.raise_objection(this, "Running stress test");

      `uvm_info("TEST", "========================================", UVM_NONE)
      `uvm_info("TEST", "=== UVM Stress Test Starting ===", UVM_NONE)
      `uvm_info("TEST", "========================================", UVM_NONE)

      // Run sequence tests on agent1
      `uvm_info("TEST", "--- Running burst sequence ---", UVM_LOW)
      burst_seq = burst_sequence::type_id::create("burst_seq");
      burst_seq.start(env.agent1.sequencer);

      // Run lock stress on agent2
      `uvm_info("TEST", "--- Running lock stress sequence ---", UVM_LOW)
      lock_seq = lock_stress_sequence::type_id::create("lock_seq");
      lock_seq.start(env.agent2.sequencer);

      // Run nested sequence
      `uvm_info("TEST", "--- Running nested sequence ---", UVM_LOW)
      nested_seq = nested_sequence::type_id::create("nested_seq");
      nested_seq.start(env.agent1.sequencer);

      // Run virtual sequence (without sub-sequencers in this basic test)
      `uvm_info("TEST", "--- Running virtual sequence ---", UVM_LOW)
      virt_seq = virtual_stress_sequence::type_id::create("virt_seq");
      virt_seq.sub_seqr1 = env.agent1.sequencer;
      virt_seq.sub_seqr2 = env.agent2.sequencer;
      virt_seq.start(null);

      #100ns;

      phase.drop_objection(this, "Stress test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      uvm_report_server server;
      int errors;

      super.report_phase(phase);

      server = uvm_report_server::get_server();
      errors = server.get_error_count();

      `uvm_info("TEST", "========================================", UVM_NONE)
      `uvm_info("TEST", "=== UVM Stress Test Results ===", UVM_NONE)
      `uvm_info("TEST", "========================================", UVM_NONE)

      // Aggregate results from stress components
      total_pass += env.cfg_stress.test_pass;
      total_fail += env.cfg_stress.test_fail;
      total_pass += env.rpt_stress.test_pass;
      total_fail += env.rpt_stress.test_fail;
      total_pass += env.ph_stress.test_pass;
      total_fail += env.ph_stress.test_fail;

      `uvm_info("TEST", $sformatf("Total Stress Tests: %0d passed, %0d failed",
                                  total_pass, total_fail), UVM_NONE)
      `uvm_info("TEST", $sformatf("Driver 1 items: %0d", env.agent1.driver.items_processed), UVM_NONE)
      `uvm_info("TEST", $sformatf("Driver 2 items: %0d", env.agent2.driver.items_processed), UVM_NONE)
      `uvm_info("TEST", $sformatf("Coverage samples: %0d", env.cov.sample_count), UVM_NONE)

      if (total_fail == 0 && errors == 0)
        `uvm_info("TEST", "*** STRESS TEST PASSED ***", UVM_NONE)
      else
        `uvm_error("TEST", $sformatf("*** STRESS TEST FAILED *** (%0d failures, %0d errors)",
                                     total_fail, errors))

      server.report_summarize();
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module uvm_stress_test_top;
  import uvm_pkg::*;
  import uvm_stress_test_pkg::*;

  // Clock for any timing-sensitive operations
  logic clk;
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    `uvm_info("TB", "Starting UVM Stress Test", UVM_NONE)
    run_test("uvm_stress_test");
  end

  // Timeout watchdog
  initial begin
    #100000;
    $display("ERROR: Test timeout!");
    $finish;
  end

endmodule
