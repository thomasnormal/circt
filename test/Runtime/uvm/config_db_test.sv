//===----------------------------------------------------------------------===//
// UVM config_db Unit Tests for CIRCT Runtime
//===----------------------------------------------------------------------===//
// Tests for the uvm_config_db implementation including:
//   - Basic set/get operations
//   - Wildcard pattern matching (* and ?)
//   - Virtual interface passing
//   - Configuration object passing
//   - Hierarchical path matching
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top config_db_test_top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM
//
// SIM: Starting config_db tests
// SIM: Tests passed:
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR
// SIM: [circt-sim] Simulation completed

`timescale 1ns/1ps

`include "uvm_macros.svh"

package config_db_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Test configuration object
  //==========================================================================
  class test_config extends uvm_object;
    `uvm_object_utils(test_config)

    int data_width = 32;
    int addr_width = 16;
    bit enable_coverage = 1;
    string name_prefix = "default";

    function new(string name = "test_config");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("data_width=%0d, addr_width=%0d, enable_coverage=%0b, name_prefix=%s",
                       data_width, addr_width, enable_coverage, name_prefix);
    endfunction
  endclass

  //==========================================================================
  // Agent configuration
  //==========================================================================
  class agent_config extends uvm_object;
    `uvm_object_utils(agent_config)

    bit is_active = 1;
    int timeout_cycles = 1000;

    function new(string name = "agent_config");
      super.new(name);
    endfunction
  endclass

  //==========================================================================
  // Test component hierarchy
  //==========================================================================
  class test_agent extends uvm_agent;
    `uvm_component_utils(test_agent)

    agent_config cfg;
    bit got_cfg_from_db;
    bit got_cfg_direct;
    bit got_cfg_fallback;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      got_cfg_from_db = 0;
      got_cfg_direct = 0;
      got_cfg_fallback = 0;
      // Try to get config from database
      if (!uvm_config_db#(agent_config)::get(this, "", "config", cfg)) begin
        string full_name_for_lookup;
        full_name_for_lookup = get_full_name();
        if (full_name_for_lookup.len() > 0 && full_name_for_lookup.substr(0, 0) == ".")
          full_name_for_lookup = full_name_for_lookup.substr(1, full_name_for_lookup.len() - 1);
        // Fallback to absolute full-name lookup.
        if (!uvm_config_db#(agent_config)::get(get_parent(), get_name(), "config", cfg) &&
            !uvm_config_db#(agent_config)::get(null, full_name_for_lookup, "config", cfg) &&
            !uvm_config_db#(agent_config)::get(null, get_full_name(), "config", cfg)) begin
          `uvm_info("AGENT", "No config found, using defaults", UVM_MEDIUM)
          cfg = agent_config::type_id::create("cfg");
        end else begin
          got_cfg_from_db = 1;
          got_cfg_fallback = 1;
          `uvm_info("AGENT", $sformatf("Got config via full_name fallback: is_active=%0b, timeout=%0d",
                                       cfg.is_active, cfg.timeout_cycles), UVM_MEDIUM)
        end
      end
      else begin
        got_cfg_from_db = 1;
        got_cfg_direct = 1;
        `uvm_info("AGENT", $sformatf("Got config: is_active=%0b, timeout=%0d",
                                     cfg.is_active, cfg.timeout_cycles), UVM_MEDIUM)
      end
    endfunction
  endclass

  class test_env extends uvm_env;
    `uvm_component_utils(test_env)

    test_agent agent1;
    test_agent agent2;
    test_config env_cfg;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Get environment config
      if (!uvm_config_db#(test_config)::get(this, "", "env_config", env_cfg)) begin
        `uvm_info("ENV", "No env config found", UVM_MEDIUM)
      end
      else begin
        `uvm_info("ENV", $sformatf("Got env config: %s", env_cfg.convert2string()), UVM_MEDIUM)
      end

      agent1 = test_agent::type_id::create("agent1", this);
      agent2 = test_agent::type_id::create("agent2", this);
    endfunction
  endclass

  //==========================================================================
  // Main Test Class
  //==========================================================================
  class config_db_test extends uvm_test;
    `uvm_component_utils(config_db_test)

    test_env env;
    int test_pass_count;
    int test_fail_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      test_pass_count = 0;
      test_fail_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Run all config_db tests before creating hierarchy
      run_uvm_is_match_tests();
      run_basic_config_db_tests();
      run_wildcard_config_db_tests();
      run_hierarchical_config_db_tests();

      // Create environment
      env = test_env::type_id::create("env", this);
    endfunction

    //========================================================================
    // Test helper
    //========================================================================
    function void check_result(bit condition, string test_name);
      if (condition) begin
        test_pass_count++;
        `uvm_info("PASS", test_name, UVM_MEDIUM)
      end
      else begin
        test_fail_count++;
        `uvm_error("FAIL", test_name)
      end
    endfunction

    //========================================================================
    // uvm_is_match tests
    //========================================================================
    function void run_uvm_is_match_tests();
      `uvm_info("TEST", "=== Running uvm_is_match tests ===", UVM_LOW)

      // Exact match
      check_result(uvm_is_match("hello", "hello"), "Exact match: hello == hello");
      check_result(!uvm_is_match("hello", "world"), "No match: hello != world");
      check_result(!uvm_is_match("hello", "hell"), "Prefix mismatch: hello != hell");
      check_result(!uvm_is_match("hell", "hello"), "Suffix mismatch: hell != hello");

      // Single wildcard *
      check_result(uvm_is_match("*", "anything"), "Wildcard * matches anything");
      check_result(uvm_is_match("*", ""), "Wildcard * matches empty string");
      check_result(uvm_is_match("hello*", "hello"), "Trailing *: hello* matches hello");
      check_result(uvm_is_match("hello*", "helloworld"), "Trailing *: hello* matches helloworld");
      check_result(uvm_is_match("*world", "helloworld"), "Leading *: *world matches helloworld");
      check_result(uvm_is_match("*world", "world"), "Leading *: *world matches world");
      check_result(uvm_is_match("hel*rld", "helloworld"), "Middle *: hel*rld matches helloworld");

      // Multiple wildcards
      check_result(uvm_is_match("*.*", "env.agent"), "Multiple *: *.* matches env.agent");
      check_result(uvm_is_match("*agent*", "env.agent.driver"), "*agent* matches env.agent.driver");
      check_result(uvm_is_match("env.*.*", "env.agent.driver"), "env.*.* matches env.agent.driver");

      // Question mark wildcard
      check_result(uvm_is_match("hel?o", "hello"), "? wildcard: hel?o matches hello");
      check_result(!uvm_is_match("hel?o", "helo"), "? wildcard: hel?o doesn't match helo");
      check_result(uvm_is_match("h?l?o", "hello"), "Multiple ?: h?l?o matches hello");

      // Combined wildcards
      check_result(uvm_is_match("h*l?o", "hello"), "Combined: h*l?o matches hello");
      check_result(uvm_is_match("h*l?o", "heeello"), "Combined: h*l?o matches heeello");

      // Edge cases
      check_result(uvm_is_match("", ""), "Empty pattern matches empty string");
      check_result(uvm_is_match("", "nonempty"), "Empty pattern matches non-empty (legacy)");
      check_result(uvm_is_match("***", "anything"), "Multiple consecutive *");

      // UVM-specific patterns
      check_result(uvm_is_match("uvm_test_top.*", "uvm_test_top.env"), "UVM path: uvm_test_top.*");
      check_result(uvm_is_match("uvm_test_top.env.*", "uvm_test_top.env.agent"), "UVM path: nested");
      check_result(uvm_is_match("*agent*", "uvm_test_top.env.my_agent.driver"), "Contains agent");
    endfunction

    //========================================================================
    // Basic config_db tests
    //========================================================================
    function void run_basic_config_db_tests();
      int int_val;
      string str_val;
      test_config cfg;

      `uvm_info("TEST", "=== Running basic config_db tests ===", UVM_LOW)

      // Test integer config
      uvm_config_db#(int)::set(null, "test_path", "int_field", 42);
      check_result(uvm_config_db#(int)::get(null, "test_path", "int_field", int_val) && int_val == 42,
            "Basic int set/get");

      // Test string config
      uvm_config_db#(string)::set(null, "test_path", "str_field", "test_value");
      check_result(uvm_config_db#(string)::get(null, "test_path", "str_field", str_val) &&
            str_val == "test_value",
            "Basic string set/get");

      // Test object config
      cfg = new("my_config");
      cfg.data_width = 64;
      cfg.addr_width = 32;
      uvm_config_db#(test_config)::set(null, "test_path", "obj_field", cfg);
      begin
        test_config retrieved_cfg;
        check_result(uvm_config_db#(test_config)::get(null, "test_path", "obj_field", retrieved_cfg) &&
              retrieved_cfg.data_width == 64 && retrieved_cfg.addr_width == 32,
              "Basic object set/get");
      end

      // Test exists
      check_result(uvm_config_db#(int)::exists(null, "test_path", "int_field"),
            "exists() returns true for existing entry");
      check_result(!uvm_config_db#(int)::exists(null, "test_path", "nonexistent"),
            "exists() returns false for nonexistent entry");

      // Test overwrite
      uvm_config_db#(int)::set(null, "test_path", "int_field", 100);
      check_result(uvm_config_db#(int)::get(null, "test_path", "int_field", int_val) && int_val == 100,
            "Overwrite existing value");
    endfunction

    //========================================================================
    // Wildcard config_db tests
    //========================================================================
    function void run_wildcard_config_db_tests();
      int int_val;
      agent_config cfg;

      `uvm_info("TEST", "=== Running wildcard config_db tests ===", UVM_LOW)

      // Test global wildcard "*"
      uvm_config_db#(int)::set(null, "*", "global_field", 999);
      check_result(uvm_config_db#(int)::get(null, "any.path.here", "global_field", int_val) &&
            int_val == 999,
            "Global * wildcard matches any path");
      check_result(uvm_config_db#(int)::get(null, "", "global_field", int_val) &&
            int_val == 999,
            "Global * wildcard matches empty path");

      // Test suffix wildcard
      cfg = new("suffix_cfg");
      cfg.is_active = 0;
      cfg.timeout_cycles = 500;
      uvm_config_db#(agent_config)::set(null, "*agent*", "config", cfg);
      begin
        agent_config retrieved_cfg;
        check_result(uvm_config_db#(agent_config)::get(null, "env.my_agent", "config", retrieved_cfg) &&
              retrieved_cfg.is_active == 0 && retrieved_cfg.timeout_cycles == 500,
              "*agent* matches env.my_agent");
        check_result(uvm_config_db#(agent_config)::get(null, "test.agent.driver", "config", retrieved_cfg),
              "*agent* matches test.agent.driver");
      end

      // Test prefix wildcard
      uvm_config_db#(int)::set(null, "env.*", "env_setting", 123);
      check_result(uvm_config_db#(int)::get(null, "env.agent", "env_setting", int_val) &&
            int_val == 123,
            "env.* matches env.agent");
      check_result(uvm_config_db#(int)::get(null, "env.scoreboard", "env_setting", int_val) &&
            int_val == 123,
            "env.* matches env.scoreboard");

      // Test exact match takes precedence over wildcard
      uvm_config_db#(int)::set(null, "*", "precedence_field", 1);
      uvm_config_db#(int)::set(null, "specific.path", "precedence_field", 2);
      check_result(uvm_config_db#(int)::get(null, "specific.path", "precedence_field", int_val) &&
            int_val == 2,
            "Exact match takes precedence over wildcard");
      check_result(uvm_config_db#(int)::get(null, "other.path", "precedence_field", int_val) &&
            int_val == 1,
            "Wildcard still matches non-specific paths");

      // Test multiple wildcards - last match wins
      uvm_config_db#(int)::set(null, "*", "multi_wild", 10);
      uvm_config_db#(int)::set(null, "env.*", "multi_wild", 20);
      uvm_config_db#(int)::set(null, "env.agent*", "multi_wild", 30);
      check_result(uvm_config_db#(int)::get(null, "env.agent1", "multi_wild", int_val) &&
            int_val == 30,
            "More specific wildcard wins (env.agent*)");
    endfunction

    //========================================================================
    // Hierarchical config_db tests
    //========================================================================
    function void run_hierarchical_config_db_tests();
      int int_val;
      test_config env_cfg;
      agent_config agent_cfg;

      `uvm_info("TEST", "=== Running hierarchical config_db tests ===", UVM_LOW)

      // Set up configuration for the test hierarchy
      // Test -> env -> agent1, agent2

      // Environment config with wildcard
      env_cfg = new("env_cfg");
      env_cfg.data_width = 128;
      env_cfg.enable_coverage = 1;
      uvm_config_db#(test_config)::set(this, "*", "env_config", env_cfg);

      // Agent configs using wildcard patterns
      agent_cfg = new("agent1_cfg");
      agent_cfg.is_active = 1;
      agent_cfg.timeout_cycles = 2000;
      uvm_config_db#(agent_config)::set(this, "env.agent1", "config", agent_cfg);

      agent_cfg = new("agent2_cfg");
      agent_cfg.is_active = 0;
      agent_cfg.timeout_cycles = 1500;
      uvm_config_db#(agent_config)::set(this, "env.agent2", "config", agent_cfg);

      // Verify hierarchical lookup from component context would work
      // Note: This tests the path construction logic
      check_result(uvm_config_db#(test_config)::exists(this, "env", "env_config"),
            "Hierarchical exists check for env_config");
    endfunction

    //========================================================================
    // Report results
    //========================================================================
    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      begin
        agent_config direct_cfg;
        check_result(uvm_config_db#(agent_config)::get(this, "env.agent1", "config", direct_cfg) &&
              direct_cfg.is_active == 1 && direct_cfg.timeout_cycles == 2000,
              "Direct config_db get from test context finds env.agent1");
        check_result(uvm_config_db#(agent_config)::get(this, "env.agent2", "config", direct_cfg) &&
              direct_cfg.is_active == 0 && direct_cfg.timeout_cycles == 1500,
              "Direct config_db get from test context finds env.agent2");
      end
      check_result(env.agent1 != null && env.agent1.got_cfg_from_db && env.agent1.cfg != null &&
            env.agent1.got_cfg_direct && !env.agent1.got_cfg_fallback &&
            env.agent1.cfg.is_active == 1 &&
            env.agent1.cfg.timeout_cycles == 2000,
            "Agent1 resolves config_db entry during build");
      check_result(env.agent2 != null && env.agent2.got_cfg_from_db && env.agent2.cfg != null &&
            env.agent2.got_cfg_direct && !env.agent2.got_cfg_fallback &&
            env.agent2.cfg.is_active == 0 &&
            env.agent2.cfg.timeout_cycles == 1500,
            "Agent2 resolves config_db entry during build");
      `uvm_info("RESULTS", $sformatf("Tests passed: %0d, Tests failed: %0d",
                                     test_pass_count, test_fail_count), UVM_NONE)
      if (test_fail_count == 0)
        `uvm_info("RESULTS", "ALL TESTS PASSED", UVM_NONE)
      else
        `uvm_error("RESULTS", "SOME TESTS FAILED")
    endfunction

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module config_db_test_top;
  import uvm_pkg::*;
  import config_db_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting config_db tests", UVM_NONE)
    run_test("config_db_test");
  end

endmodule
