//===----------------------------------------------------------------------===//
// UVM Package Stubs for CIRCT/Slang Compilation
//===----------------------------------------------------------------------===//
//
// This file provides minimal UVM class and type definitions that allow
// testbench code using UVM to compile with circt-verilog. These are stubs -
// they provide the interface but not the full UVM functionality.
//
// Supported classes:
//   - uvm_void, uvm_object, uvm_transaction, uvm_sequence_item
//   - uvm_component, uvm_agent, uvm_driver, uvm_monitor
//   - uvm_sequencer, uvm_sequencer_base, uvm_sequence, uvm_sequence_base
//   - uvm_env, uvm_test, uvm_scoreboard, uvm_subscriber
//   - uvm_phase, uvm_config_db, uvm_factory
//   - TLM: uvm_analysis_port, uvm_analysis_imp, uvm_tlm_analysis_fifo, etc.
//   - Comparators: uvm_in_order_comparator, uvm_in_order_built_in_comparator,
//                  uvm_algorithmic_comparator, uvm_pair
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`ifndef UVM_PKG_SV
`define UVM_PKG_SV

// Include UVM macros before the package so they're available
`include "uvm_macros.svh"

package uvm_pkg;

  //=========================================================================
  // Forward Declarations
  //=========================================================================
  typedef class uvm_object;
  typedef class uvm_component;
  typedef class uvm_sequencer_base;
  typedef class uvm_sequence_base;
  typedef class uvm_phase;
  typedef class uvm_factory;
  typedef class uvm_root;
  typedef class uvm_comparer;
  typedef class uvm_printer;
  typedef class uvm_packer;
  typedef class uvm_recorder;
  typedef class uvm_objection;
  typedef class uvm_seq_item_pull_port;
  typedef class uvm_seq_item_pull_imp;
  typedef class uvm_put_imp;
  typedef class uvm_get_peek_imp;
  typedef class uvm_analysis_port;
  typedef class uvm_analysis_imp;
  typedef class uvm_subscriber;
  typedef class uvm_tlm_fifo;
  typedef class uvm_reg_map;
  typedef class uvm_reg;
  typedef class uvm_reg_field;
  typedef class uvm_mem;
  typedef class uvm_component_registry;
  typedef class uvm_object_wrapper;
  typedef class uvm_tr_database;
  typedef class uvm_report_server;
  typedef class uvm_report_handler;
  typedef class uvm_report_message;
  typedef class uvm_resource_base;
  typedef class uvm_resource_pool;
  typedef class uvm_default_coreservice_t;
  typedef class uvm_coverage;
  typedef class uvm_mem_region;
  typedef class uvm_mem_mam;
  typedef class uvm_mem_mam_cfg;
  typedef class uvm_coverage_db;

  typedef int unsigned uvm_bitstream_t;
  typedef longint uvm_integral_t;
  typedef bit [63:0] uvm_reg_addr_t;
  typedef bit [63:0] uvm_reg_data_t;
  typedef bit [7:0] uvm_reg_byte_en_t;

  typedef enum {
    UVM_READ,
    UVM_WRITE,
    UVM_BURST_READ,
    UVM_BURST_WRITE
  } uvm_access_e;

  typedef enum {
    UVM_IS_OK,
    UVM_NOT_OK,
    UVM_HAS_X
  } uvm_status_e;

  typedef struct {
    uvm_access_e kind;
    uvm_reg_addr_t addr;
    uvm_reg_data_t data;
    int n_bits;
    uvm_reg_byte_en_t byte_en;
    uvm_status_e status;
  } uvm_reg_bus_op;

  typedef enum {
    UVM_FRONTDOOR,
    UVM_BACKDOOR,
    UVM_PREDICT,
    UVM_DEFAULT_PATH
  } uvm_path_e;

  typedef enum {
    UVM_APPEND,
    UVM_PREPEND
  } uvm_apprepend;

  typedef enum {
    UVM_ALL_ACTIVE,
    UVM_ONE_ACTIVE,
    UVM_ANY_ACTIVE,
    UVM_NO_HB_MODE
  } uvm_heartbeat_modes;

  //=========================================================================
  // Enumerations
  //=========================================================================

  // UVM verbosity levels
  typedef enum int {
    UVM_NONE   = 0,
    UVM_LOW    = 100,
    UVM_MEDIUM = 200,
    UVM_HIGH   = 300,
    UVM_FULL   = 400,
    UVM_DEBUG  = 500
  } uvm_verbosity;

  // UVM severity levels
  typedef enum bit [1:0] {
    UVM_INFO    = 2'b00,
    UVM_WARNING = 2'b01,
    UVM_ERROR   = 2'b10,
    UVM_FATAL   = 2'b11
  } uvm_severity;

  // UVM action type
  typedef enum int {
    UVM_NO_ACTION      = 'h00,
    UVM_DISPLAY        = 'h01,
    UVM_LOG            = 'h02,
    UVM_COUNT          = 'h04,
    UVM_EXIT           = 'h08,
    UVM_CALL_HOOK      = 'h10,
    UVM_STOP           = 'h20,
    UVM_RM_RECORD      = 'h40
  } uvm_action_type;

  // uvm_action is a bitfield of action flags (can combine multiple actions)
  typedef int uvm_action;

  // UVM_FILE is a file handle type
  typedef int UVM_FILE;

  // Active/passive enum for agents
  typedef enum bit {
    UVM_PASSIVE = 0,
    UVM_ACTIVE  = 1
  } uvm_active_passive_enum;

  // Sequencer arbitration mode
  typedef enum int {
    UVM_SEQ_ARB_FIFO,
    UVM_SEQ_ARB_WEIGHTED,
    UVM_SEQ_ARB_RANDOM,
    UVM_SEQ_ARB_STRICT_FIFO,
    UVM_SEQ_ARB_STRICT_RANDOM,
    UVM_SEQ_ARB_USER
  } uvm_sequencer_arb_mode;

  // Object/component flags
  typedef enum int {
    UVM_COPY       = 'h001,
    UVM_NOCOPY     = 'h002,
    UVM_COMPARE    = 'h004,
    UVM_NOCOMPARE  = 'h008,
    UVM_PRINT      = 'h010,
    UVM_NOPRINT    = 'h020,
    UVM_RECORD     = 'h040,
    UVM_NORECORD   = 'h080,
    UVM_PACK       = 'h100,
    UVM_NOPACK     = 'h200,
    UVM_REFERENCE  = 'h400,
    UVM_READONLY   = 'h800,
    UVM_ALL_ON     = 'hFFF,
    UVM_DEFAULT    = 'h555
  } uvm_field_auto_enum;

  // Objection-related enums
  typedef enum {
    UVM_RAISED,
    UVM_DROPPED,
    UVM_ALL_DROPPED
  } uvm_objection_event;

  // Phase state enum (for wait_for_state)
  typedef enum {
    UVM_PHASE_DORMANT,
    UVM_PHASE_SCHEDULED,
    UVM_PHASE_SYNCING,
    UVM_PHASE_STARTED,
    UVM_PHASE_EXECUTING,
    UVM_PHASE_READY_TO_END,
    UVM_PHASE_ENDED,
    UVM_PHASE_CLEANUP,
    UVM_PHASE_DONE,
    UVM_PHASE_JUMPING
  } uvm_phase_state;

  // Port types
  typedef enum int {
    UVM_PORT,
    UVM_EXPORT,
    UVM_IMPLEMENTATION
  } uvm_port_type_e;

  // Radix for printing
  typedef enum {
    UVM_BIN    = 'h1,
    UVM_DEC    = 'h2,
    UVM_UNSIGNED = 'h3,
    UVM_OCT    = 'h4,
    UVM_HEX    = 'h5,
    UVM_STRING = 'h6,
    UVM_TIME   = 'h7,
    UVM_ENUM   = 'h8,
    UVM_REAL   = 'h9,
    UVM_REAL_DEC = 'hA,
    UVM_REAL_EXP = 'hB,
    UVM_NORADIX  = 'h0
  } uvm_radix_enum;

  //=========================================================================
  // Global Functions (Stubs)
  //=========================================================================

  // Stub for report_enabled check
  function automatic bit uvm_report_enabled(int verbosity, uvm_severity severity, string id);
    return 1; // Always enabled in stub
  endfunction

  // Stub report functions
  function automatic void uvm_report_info(string id, string msg, int verbosity = UVM_MEDIUM,
                                          string filename = "", int line = 0);
    $display("[UVM_INFO @ %0t] %s: %s", $time, id, msg);
  endfunction

  function automatic void uvm_report_warning(string id, string msg, int verbosity = UVM_NONE,
                                             string filename = "", int line = 0);
    $display("[UVM_WARNING @ %0t] %s: %s", $time, id, msg);
  endfunction

  function automatic void uvm_report_error(string id, string msg, int verbosity = UVM_NONE,
                                           string filename = "", int line = 0);
    $display("[UVM_ERROR @ %0t] %s: %s", $time, id, msg);
  endfunction

  function automatic void uvm_report_fatal(string id, string msg, int verbosity = UVM_NONE,
                                           string filename = "", int line = 0);
    $display("[UVM_FATAL @ %0t] %s: %s", $time, id, msg);
  endfunction

  // Run test task - main UVM entry point
  // This task creates the test from factory and executes all UVM phases
  // Note: We call uvm_root::get() directly rather than using uvm_top global
  // because static variable initializers with function calls are not yet
  // fully supported in CIRCT's ImportVerilog conversion.
  task automatic run_test(string test_name = "");
    uvm_root::get().run_test(test_name);
  endtask

  //=========================================================================
  // uvm_is_match - Glob pattern matching for config_db
  //=========================================================================
  // Matches a glob-style pattern against a string.
  // Supports:
  //   * - matches any sequence of characters (including empty)
  //   ? - matches any single character
  // Returns 1 if match succeeds, 0 otherwise.
  function automatic bit uvm_is_match(string expr, string str);
    int e, es, s, ss;
    e  = 0; s  = 0;
    es = 0; ss = 0;

    // Empty pattern matches everything
    if (expr.len() == 0)
      return 1;

    // Strip leading ^ if present (legacy behavior)
    if (expr[0] == "^")
      expr = expr.substr(1, expr.len()-1);

    // Match non-wildcard prefix
    while (s != str.len() && e < expr.len() && expr.getc(e) != "*") begin
      if ((expr.getc(e) != str.getc(s)) && (expr.getc(e) != "?"))
        return 0;
      e++; s++;
    end

    // If we've consumed the entire pattern
    if (e == expr.len())
      return (s == str.len());

    // Process remaining with wildcard matching
    while (s != str.len()) begin
      if (e < expr.len() && expr.getc(e) == "*") begin
        e++;
        if (e == expr.len())
          return 1;  // Trailing * matches everything
        es = e;
        ss = s + 1;
      end
      else if (e < expr.len() && (expr.getc(e) == str.getc(s) || expr.getc(e) == "?")) begin
        e++;
        s++;
      end
      else begin
        e = es;
        s = ss++;
      end
    end

    // Skip trailing wildcards
    while (e < expr.len() && expr.getc(e) == "*")
      e++;

    return (e == expr.len());
  endfunction

  // uvm_re_match - Returns 0 on match, 1 on no-match (like C strcmp convention)
  function automatic int uvm_re_match(string re, string str);
    return uvm_is_match(re, str) ? 0 : 1;
  endfunction

  // uvm_glob_to_re - Converts glob to regex (stub - just returns glob)
  function automatic string uvm_glob_to_re(string glob);
    return glob;
  endfunction

  //=========================================================================
  // uvm_void - Root class
  //=========================================================================
  virtual class uvm_void;
  endclass

  //=========================================================================
  // uvm_object - Base class for data objects
  //=========================================================================
  class uvm_object extends uvm_void;
    protected string m_name;
    local int m_inst_id;
    static local int m_inst_count = 0;

    function new(string name = "");
      m_name = name;
      m_inst_id = m_inst_count++;
    endfunction

    virtual function string get_name();
      return m_name;
    endfunction

    virtual function void set_name(string name);
      m_name = name;
    endfunction

    virtual function string get_full_name();
      return m_name;
    endfunction

    virtual function string get_type_name();
      return "uvm_object";
    endfunction

    virtual function int get_inst_id();
      return m_inst_id;
    endfunction

    // Clone and copy
    virtual function uvm_object clone();
      return null; // Override in derived classes
    endfunction

    virtual function void copy(uvm_object rhs);
      if (rhs != null)
        m_name = rhs.m_name;
    endfunction

    virtual function bit compare(uvm_object rhs, uvm_comparer comparer = null);
      return 1; // Stub always returns match
    endfunction

    // Print and sprint
    virtual function void print(uvm_printer printer = null);
      $display("%s", sprint(printer));
    endfunction

    virtual function string sprint(uvm_printer printer = null);
      return $sformatf("%s: %s", get_type_name(), get_name());
    endfunction

    virtual function string convert2string();
      return sprint();
    endfunction

    // Pack/unpack
    virtual function int pack(ref bit bitstream[$],
                              input uvm_packer packer = null);
      return 0;
    endfunction

    virtual function int unpack(ref bit bitstream[$],
                                input uvm_packer packer = null);
      return 0;
    endfunction

    virtual function int pack_bytes(ref byte unsigned bytestream[], input uvm_packer packer = null);
      return 0;
    endfunction

    virtual function int unpack_bytes(ref byte unsigned bytestream[], input uvm_packer packer = null);
      return 0;
    endfunction

    virtual function int pack_ints(ref int unsigned intstream[], input uvm_packer packer = null);
      return 0;
    endfunction

    virtual function int unpack_ints(ref int unsigned intstream[], input uvm_packer packer = null);
      return 0;
    endfunction

    // Record
    virtual function void record(uvm_recorder recorder = null);
    endfunction

    // Do hooks for overriding
    virtual function void do_print(uvm_printer printer);
    endfunction

    virtual function void do_copy(uvm_object rhs);
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      return 1;
    endfunction

    virtual function void do_pack(uvm_packer packer);
    endfunction

    virtual function void do_unpack(uvm_packer packer);
    endfunction

    virtual function void do_record(uvm_recorder recorder);
    endfunction

    // Create - factory pattern
    virtual function uvm_object create(string name = "");
      return null; // Override in derived classes
    endfunction

  endclass

  //=========================================================================
  // uvm_transaction - Base class for transactions
  //=========================================================================
  class uvm_transaction extends uvm_object;
    protected longint accept_time = -1;
    protected longint begin_time = -1;
    protected longint end_time = -1;
    protected int m_transaction_id = -1;
    protected uvm_component initiator;

    function new(string name = "", uvm_component initiator = null);
      super.new(name);
      this.initiator = initiator;
    endfunction

    virtual function void set_initiator(uvm_component initiator);
      this.initiator = initiator;
    endfunction

    virtual function uvm_component get_initiator();
      return initiator;
    endfunction

    virtual function void accept_tr(longint accept_time = 0);
      this.accept_time = accept_time;
    endfunction

    virtual function int begin_tr(longint begin_time = 0);
      this.begin_time = begin_time;
      return m_transaction_id;
    endfunction

    virtual function void end_tr(longint end_time = 0, bit free_handle = 1);
      this.end_time = end_time;
    endfunction

    virtual function longint get_accept_time();
      return accept_time;
    endfunction

    virtual function longint get_begin_time();
      return begin_time;
    endfunction

    virtual function longint get_end_time();
      return end_time;
    endfunction

    virtual function int get_transaction_id();
      return m_transaction_id;
    endfunction

    virtual function void set_transaction_id(int id);
      m_transaction_id = id;
    endfunction

    virtual function string get_type_name();
      return "uvm_transaction";
    endfunction

    virtual function string convert2string();
      return $sformatf("%s [%0d]", get_name(), m_transaction_id);
    endfunction

  endclass

  //=========================================================================
  // uvm_sequence_item - Base class for sequence items
  //=========================================================================
  class uvm_sequence_item extends uvm_transaction;
    protected int m_sequence_id = -1;
    protected bit m_use_sequence_info;
    protected int m_depth = -1;
    protected uvm_sequencer_base m_sequencer;
    protected uvm_sequence_base m_parent_sequence;

    function new(string name = "uvm_sequence_item");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequence_item";
    endfunction

    virtual function void set_item_context(uvm_sequence_base parent_seq,
                                           uvm_sequencer_base sequencer = null);
      m_parent_sequence = parent_seq;
      if (sequencer != null)
        m_sequencer = sequencer;
    endfunction

    virtual function uvm_sequence_base get_parent_sequence();
      return m_parent_sequence;
    endfunction

    virtual function void set_parent_sequence(uvm_sequence_base parent);
      m_parent_sequence = parent;
    endfunction

    virtual function int get_sequence_id();
      return m_sequence_id;
    endfunction

    virtual function void set_sequencer(uvm_sequencer_base sequencer);
      m_sequencer = sequencer;
    endfunction

    virtual function uvm_sequencer_base get_sequencer();
      return m_sequencer;
    endfunction

    virtual function void set_depth(int value);
      m_depth = value;
    endfunction

    virtual function int get_depth();
      return m_depth;
    endfunction

    virtual function bit is_item();
      return 1;
    endfunction

    virtual function void set_id_info(uvm_sequence_item item);
      if (item != null) begin
        m_sequence_id = item.m_sequence_id;
        m_use_sequence_info = item.m_use_sequence_info;
      end
    endfunction

    virtual function void set_use_sequence_info(bit value);
      m_use_sequence_info = value;
    endfunction

    virtual function bit get_use_sequence_info();
      return m_use_sequence_info;
    endfunction

  endclass

  //=========================================================================
  // uvm_phase - Phase management
  //=========================================================================
  class uvm_phase extends uvm_object;
    protected string m_phase_name;
    local uvm_phase m_imp;
    local uvm_phase m_parent;
    local uvm_phase m_predecessors[$];
    local uvm_phase m_successors[$];
    local uvm_objection m_phase_objection;
    local uvm_phase_state m_state;

    function new(string name = "uvm_phase");
      super.new(name);
      m_phase_name = name;
      m_phase_objection = new({name, "_objection"});
      m_state = UVM_PHASE_DORMANT;
    endfunction

    virtual function string get_name();
      return m_phase_name;
    endfunction

    virtual function string get_full_name();
      return m_phase_name;
    endfunction

    virtual function string get_type_name();
      return "uvm_phase";
    endfunction

    // Objection methods - delegate to internal objection
    virtual function void raise_objection(uvm_object obj = null, string description = "", int count = 1);
      if (m_phase_objection != null)
        m_phase_objection.raise_objection(obj, description, count);
    endfunction

    virtual function void drop_objection(uvm_object obj = null, string description = "", int count = 1);
      if (m_phase_objection != null)
        m_phase_objection.drop_objection(obj, description, count);
    endfunction

    virtual function uvm_objection get_objection();
      return m_phase_objection;
    endfunction

    // Get objection count for this phase
    virtual function int get_objection_count(uvm_object obj = null);
      if (m_phase_objection != null)
        return m_phase_objection.get_objection_count(obj);
      return 0;
    endfunction

    // Phase state
    virtual function bit is_before(uvm_phase phase);
      return 0;
    endfunction

    virtual function bit is_after(uvm_phase phase);
      return 0;
    endfunction

    // Get current phase state
    virtual function uvm_phase_state get_state();
      return m_state;
    endfunction

    // Set phase state (internal use)
    virtual function void set_state(uvm_phase_state state);
      m_state = state;
    endfunction

    // Wait for phase to reach specified state
    // In stub implementation, we assume the phase is already in the requested
    // state or will transition immediately (since we don't have a real phase machine)
    virtual task wait_for_state(uvm_phase_state state, uvm_objection_event op = UVM_ALL_DROPPED);
      // Stub: In a real implementation, this would block until the phase
      // reaches the specified state. For compilation purposes, we set the
      // state immediately and return.
      m_state = state;
      #0; // Allow other processes to execute
    endtask

  endclass

  // Common phase instances (global)
  uvm_phase build_phase_h = new("build");
  uvm_phase connect_phase_h = new("connect");
  uvm_phase end_of_elaboration_phase_h = new("end_of_elaboration");
  uvm_phase start_of_simulation_phase_h = new("start_of_simulation");
  uvm_phase run_phase_h = new("run");
  uvm_phase extract_phase_h = new("extract");
  uvm_phase check_phase_h = new("check");
  uvm_phase report_phase_h = new("report");
  uvm_phase final_phase_h = new("final");

  // Standard UVM phase handle aliases (with _ph suffix per IEEE 1800.2)
  // These are the names used in real-world UVM code
  uvm_phase build_ph = build_phase_h;
  uvm_phase connect_ph = connect_phase_h;
  uvm_phase end_of_elaboration_ph = end_of_elaboration_phase_h;
  uvm_phase start_of_simulation_ph = start_of_simulation_phase_h;
  uvm_phase run_ph = run_phase_h;
  uvm_phase extract_ph = extract_phase_h;
  uvm_phase check_ph = check_phase_h;
  uvm_phase report_ph = report_phase_h;
  uvm_phase final_ph = final_phase_h;

  //=========================================================================
  // uvm_objection - Objection mechanism
  //=========================================================================
  class uvm_objection extends uvm_object;
    protected int m_count;
    protected int m_total_count;
    protected time m_drain_time;

    function new(string name = "uvm_objection");
      super.new(name);
      m_count = 0;
      m_total_count = 0;
      m_drain_time = 0;
    endfunction

    virtual function void raise_objection(uvm_object obj = null, string description = "", int count = 1);
      m_count += count;
      m_total_count += count;
    endfunction

    virtual function void drop_objection(uvm_object obj = null, string description = "", int count = 1);
      m_count -= count;
      if (m_count < 0)
        m_count = 0;
    endfunction

    virtual function void set_drain_time(uvm_object obj = null, time drain);
      m_drain_time = drain;
    endfunction

    virtual function time get_drain_time(uvm_object obj = null);
      return m_drain_time;
    endfunction

    virtual function int get_objection_count(uvm_object obj = null);
      return m_count;
    endfunction

    // Get the total count of objections ever raised
    virtual function int get_objection_total(uvm_object obj = null);
      return m_total_count;
    endfunction

    // Clear all objections
    virtual function void clear(uvm_object obj = null);
      m_count = 0;
    endfunction

    // Check if there are any raised objections
    virtual function bit raised();
      return m_count > 0;
    endfunction

    // Wait for a specific objection event
    virtual task wait_for(uvm_objection_event objt_event, uvm_object obj = null);
      case (objt_event)
        UVM_ALL_DROPPED: begin
          while (m_count > 0)
            #1;
        end
        default: begin
          // Other events are not currently simulated
        end
      endcase
    endtask

    // Display objection state
    virtual function void display_objections(uvm_object obj = null, bit show_header = 1);
      if (show_header)
        $display("Objection report for '%s':", get_name());
      $display("  Count: %0d", m_count);
      $display("  Total raised: %0d", m_total_count);
    endfunction

  endclass

  //=========================================================================
  // uvm_test_done_objection - Global test completion objection
  //=========================================================================
  class uvm_test_done_objection extends uvm_objection;
    local static uvm_test_done_objection m_inst;

    function new(string name = "uvm_test_done");
      super.new(name);
    endfunction

    static function uvm_test_done_objection get();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

  endclass

  uvm_test_done_objection uvm_test_done = uvm_test_done_objection::get();

  //=========================================================================
  // uvm_report_object - Reporting base class
  //=========================================================================
  class uvm_report_object extends uvm_object;
    protected int m_verbosity = UVM_MEDIUM;
    protected uvm_report_handler m_rh;

    function new(string name = "");
      super.new(name);
    endfunction

    // Get the report handler for this object
    virtual function uvm_report_handler get_report_handler();
      if (m_rh == null)
        m_rh = new("report_handler");
      return m_rh;
    endfunction

    // Set a custom report handler
    virtual function void set_report_handler(uvm_report_handler handler);
      m_rh = handler;
    endfunction

    virtual function void uvm_report_info(string id, string message,
                                          int verbosity = UVM_MEDIUM,
                                          string filename = "", int line = 0);
      uvm_report_server server = uvm_report_server::get_server();
      if (verbosity <= m_verbosity) begin
        if (report_hook(id, message, verbosity, filename, line)) begin
          $display("[%0t] UVM_INFO %s (%s) %s", $time, get_full_name(), id, message);
          server.incr_severity_count(UVM_INFO);
          server.incr_id_count(id);
        end
      end
    endfunction

    virtual function void uvm_report_warning(string id, string message,
                                             int verbosity = UVM_MEDIUM,
                                             string filename = "", int line = 0);
      uvm_report_server server = uvm_report_server::get_server();
      if (report_hook(id, message, verbosity, filename, line)) begin
        $display("[%0t] UVM_WARNING %s (%s) %s", $time, get_full_name(), id, message);
        server.incr_severity_count(UVM_WARNING);
        server.incr_id_count(id);
      end
    endfunction

    virtual function void uvm_report_error(string id, string message,
                                           int verbosity = UVM_LOW,
                                           string filename = "", int line = 0);
      uvm_report_server server = uvm_report_server::get_server();
      if (report_hook(id, message, verbosity, filename, line)) begin
        $display("[%0t] UVM_ERROR %s (%s) %s", $time, get_full_name(), id, message);
        server.incr_severity_count(UVM_ERROR);
        server.incr_id_count(id);
        server.incr_quit_count();
      end
    endfunction

    virtual function void uvm_report_fatal(string id, string message,
                                           int verbosity = UVM_NONE,
                                           string filename = "", int line = 0);
      uvm_report_server server = uvm_report_server::get_server();
      if (report_hook(id, message, verbosity, filename, line)) begin
        $display("[%0t] UVM_FATAL %s (%s) %s", $time, get_full_name(), id, message);
        server.incr_severity_count(UVM_FATAL);
        server.incr_id_count(id);
      end
    endfunction

    virtual function void set_report_verbosity_level(int verbosity);
      m_verbosity = verbosity;
    endfunction

    virtual function int get_report_verbosity_level();
      return m_verbosity;
    endfunction

    // Report hook - override to customize report handling
    // Return 1 to allow message, 0 to suppress
    virtual function bit report_hook(
      string id,
      string message,
      int verbosity,
      string filename,
      int line
    );
      // Default: allow all messages
      return 1;
    endfunction

    // Set max quit count on the report server
    virtual function void set_report_max_quit_count(int count);
      uvm_report_server server = uvm_report_server::get_server();
      server.set_max_quit_count(count);
    endfunction

    // Get max quit count from the report server
    virtual function int get_report_max_quit_count();
      uvm_report_server server = uvm_report_server::get_server();
      return server.get_max_quit_count();
    endfunction

    // Action configuration methods
    virtual function void set_report_severity_action(uvm_severity severity, uvm_action action);
      uvm_report_handler rh = get_report_handler();
      rh.set_severity_action(severity, action);
    endfunction

    virtual function void set_report_id_action(string id, uvm_action action);
      uvm_report_handler rh = get_report_handler();
      rh.set_id_action(id, action);
    endfunction

    virtual function void set_report_severity_id_action(uvm_severity severity, string id, uvm_action action);
      uvm_report_handler rh = get_report_handler();
      rh.set_severity_id_action(severity, id, action);
    endfunction

    // Verbosity configuration methods
    virtual function void set_report_id_verbosity(string id, int verbosity);
      uvm_report_handler rh = get_report_handler();
      rh.set_id_verbosity(id, verbosity);
    endfunction

    virtual function void set_report_severity_id_verbosity(uvm_severity severity, string id, int verbosity);
      uvm_report_handler rh = get_report_handler();
      rh.set_severity_id_verbosity(severity, id, verbosity);
    endfunction

    // File handle configuration methods
    virtual function void set_report_default_file(UVM_FILE file);
      uvm_report_handler rh = get_report_handler();
      rh.set_default_file(file);
    endfunction

    virtual function void set_report_severity_file(uvm_severity severity, UVM_FILE file);
      uvm_report_handler rh = get_report_handler();
      rh.set_severity_file(severity, file);
    endfunction

    virtual function void set_report_id_file(string id, UVM_FILE file);
      uvm_report_handler rh = get_report_handler();
      rh.set_id_file(id, file);
    endfunction

    virtual function void set_report_severity_id_file(uvm_severity severity, string id, UVM_FILE file);
      uvm_report_handler rh = get_report_handler();
      rh.set_severity_id_file(severity, id, file);
    endfunction

    // Returns this object as a report object (compatibility with real UVM)
    virtual function uvm_report_object uvm_get_report_object();
      return this;
    endfunction

  endclass

  //=========================================================================
  // uvm_component - Base class for all components
  //=========================================================================
  class uvm_component extends uvm_report_object;
    protected uvm_component m_parent;
    protected uvm_component m_children[string];
    protected uvm_phase m_current_phase;
    protected bit m_build_done;
    local string m_name;

    function new(string name, uvm_component parent);
      super.new(name);
      m_name = name;
      m_parent = parent;
      m_build_done = 0;
      if (parent != null)
        parent.m_children[name] = this;
    endfunction

    virtual function string get_name();
      return m_name;
    endfunction

    virtual function string get_full_name();
      if (m_parent == null || m_parent.get_name() == "")
        return m_name;
      else
        return {m_parent.get_full_name(), ".", m_name};
    endfunction

    virtual function string get_type_name();
      return "uvm_component";
    endfunction

    virtual function uvm_component get_parent();
      return m_parent;
    endfunction

    virtual function int get_num_children();
      return m_children.num();
    endfunction

    virtual function int get_first_child(ref string name);
      return m_children.first(name);
    endfunction

    virtual function int get_next_child(ref string name);
      return m_children.next(name);
    endfunction

    virtual function uvm_component get_child(string name);
      if (m_children.exists(name))
        return m_children[name];
      return null;
    endfunction

    virtual function void get_children(ref uvm_component children[$]);
      foreach (m_children[name])
        children.push_back(m_children[name]);
    endfunction

    virtual function uvm_component lookup(string name);
      // Simple lookup - just returns child if exists
      return get_child(name);
    endfunction

    // Phase methods (stubs)
    virtual function void build_phase(uvm_phase phase);
      m_build_done = 1;
    endfunction

    virtual function void connect_phase(uvm_phase phase);
    endfunction

    virtual function void end_of_elaboration_phase(uvm_phase phase);
    endfunction

    virtual function void start_of_simulation_phase(uvm_phase phase);
    endfunction

    virtual task run_phase(uvm_phase phase);
    endtask

    virtual function void extract_phase(uvm_phase phase);
    endfunction

    virtual function void check_phase(uvm_phase phase);
    endfunction

    virtual function void report_phase(uvm_phase phase);
    endfunction

    virtual function void final_phase(uvm_phase phase);
    endfunction

    // Pre/post phase hooks
    virtual function void pre_build_phase(uvm_phase phase);
    endfunction

    virtual function void post_build_phase(uvm_phase phase);
    endfunction

    virtual function void pre_connect_phase(uvm_phase phase);
    endfunction

    virtual function void post_connect_phase(uvm_phase phase);
    endfunction

    // Run phase sub-phases
    virtual task pre_reset_phase(uvm_phase phase);
    endtask

    virtual task reset_phase(uvm_phase phase);
    endtask

    virtual task post_reset_phase(uvm_phase phase);
    endtask

    virtual task pre_configure_phase(uvm_phase phase);
    endtask

    virtual task configure_phase(uvm_phase phase);
    endtask

    virtual task post_configure_phase(uvm_phase phase);
    endtask

    virtual task pre_main_phase(uvm_phase phase);
    endtask

    virtual task main_phase(uvm_phase phase);
    endtask

    virtual task post_main_phase(uvm_phase phase);
    endtask

    virtual task pre_shutdown_phase(uvm_phase phase);
    endtask

    virtual task shutdown_phase(uvm_phase phase);
    endtask

    virtual task post_shutdown_phase(uvm_phase phase);
    endtask

    // Phase control
    virtual function uvm_phase get_current_phase();
      return m_current_phase;
    endfunction

    virtual function void set_current_phase(uvm_phase phase);
      m_current_phase = phase;
    endfunction

    // Phase objections
    virtual function void raise_objection(uvm_phase phase = null, string description = "", int count = 1);
      if (phase != null)
        phase.raise_objection(this, description, count);
    endfunction

    virtual function void drop_objection(uvm_phase phase = null, string description = "", int count = 1);
      if (phase != null)
        phase.drop_objection(this, description, count);
    endfunction

    // Topology
    virtual function void print_topology(uvm_printer printer = null);
      $display("Component: %s (%s)", get_full_name(), get_type_name());
      foreach (m_children[name])
        m_children[name].print_topology(printer);
    endfunction

    virtual function void set_report_id_verbosity(string id, int verbosity);
    endfunction

    // Factory override convenience methods (delegate to factory)
    virtual function void set_inst_override_by_type(string relative_inst_path,
                                                    uvm_object_wrapper original_type,
                                                    uvm_object_wrapper override_type);
      string full_path;
      if (get_full_name() != "")
        full_path = {get_full_name(), ".", relative_inst_path};
      else
        full_path = relative_inst_path;
      factory.set_inst_override_by_type(original_type, override_type, full_path);
    endfunction

    virtual function void set_type_override_by_type(uvm_object_wrapper original_type,
                                                    uvm_object_wrapper override_type,
                                                    bit replace = 1);
      factory.set_type_override_by_type(original_type, override_type, replace);
    endfunction

  endclass

  //=========================================================================
  // uvm_root - Top of component hierarchy
  //=========================================================================
  class uvm_root extends uvm_component;
    local static uvm_root m_inst;
    protected uvm_factory m_factory;
    protected uvm_component m_test_top;  // The test instance

    function new();
      super.new("", null);
    endfunction

    static function uvm_root get();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

    virtual function void print_topology(uvm_printer printer = null);
      $display("UVM Topology:");
      super.print_topology(printer);
    endfunction

    //-----------------------------------------------------------------------
    // Phase traversal helpers
    //-----------------------------------------------------------------------

    // Execute a function phase on a component and all its children (top-down)
    protected function void execute_phase_func(uvm_component comp, uvm_phase phase,
                                               string phase_name);
      uvm_component children[$];
      comp.set_current_phase(phase);

      // Execute phase on this component first (top-down)
      case (phase_name)
        "build": comp.build_phase(phase);
        "connect": comp.connect_phase(phase);
        "end_of_elaboration": comp.end_of_elaboration_phase(phase);
        "start_of_simulation": comp.start_of_simulation_phase(phase);
        "extract": comp.extract_phase(phase);
        "check": comp.check_phase(phase);
        "report": comp.report_phase(phase);
        "final": comp.final_phase(phase);
      endcase

      // Then recurse into children
      comp.get_children(children);
      foreach (children[i])
        execute_phase_func(children[i], phase, phase_name);
    endfunction

    // Execute run_phase on a component and all its children concurrently
    protected task execute_run_phase(uvm_component comp, uvm_phase phase);
      uvm_component children[$];
      comp.set_current_phase(phase);
      comp.get_children(children);

      // Fork run_phase for this component and all children
      fork
        comp.run_phase(phase);
      join_none

      foreach (children[i])
        execute_run_phase(children[i], phase);
    endtask

    // Wait for all objections to be dropped for a phase
    protected task wait_for_objections(uvm_phase phase);
      // Poll objection count until all dropped
      while (phase.get_objection_count() > 0) begin
        #1;  // Small time step to allow other processes
      end
    endtask

    //-----------------------------------------------------------------------
    // run_test - Main UVM entry point
    //-----------------------------------------------------------------------
    virtual task run_test(string test_name = "");
      uvm_factory f;
      uvm_component test_comp;

      `uvm_info("UVM/RUNTST", $sformatf("run_test called with: %s", test_name), UVM_LOW)

      // Get factory
      f = uvm_factory::get();

      // Create the test instance via factory
      if (test_name != "") begin
        test_comp = f.create_component_by_name(test_name, "", "uvm_test_top", this);
        if (test_comp == null) begin
          `uvm_fatal("UVM/RUNTST", $sformatf("Cannot find test '%s' - not registered with factory", test_name))
          return;
        end
        m_test_top = test_comp;
        `uvm_info("UVM/RUNTST", $sformatf("Created test: %s", test_comp.get_type_name()), UVM_LOW)
      end
      else begin
        `uvm_warning("UVM/RUNTST", "No test name specified")
        return;
      end

      //---------------------------------------------------------------------
      // Execute UVM phases in order
      //---------------------------------------------------------------------

      // BUILD PHASE (top-down: parent builds children)
      `uvm_info("UVM/PHASE", "Starting build_phase", UVM_LOW)
      build_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, build_phase_h, "build");
      build_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed build_phase", UVM_LOW)

      // CONNECT PHASE
      `uvm_info("UVM/PHASE", "Starting connect_phase", UVM_LOW)
      connect_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, connect_phase_h, "connect");
      connect_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed connect_phase", UVM_LOW)

      // END_OF_ELABORATION PHASE
      `uvm_info("UVM/PHASE", "Starting end_of_elaboration_phase", UVM_LOW)
      end_of_elaboration_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, end_of_elaboration_phase_h, "end_of_elaboration");
      end_of_elaboration_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed end_of_elaboration_phase", UVM_LOW)

      // START_OF_SIMULATION PHASE
      `uvm_info("UVM/PHASE", "Starting start_of_simulation_phase", UVM_LOW)
      start_of_simulation_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, start_of_simulation_phase_h, "start_of_simulation");
      start_of_simulation_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed start_of_simulation_phase", UVM_LOW)

      // RUN PHASE (concurrent execution with objection handling)
      `uvm_info("UVM/PHASE", "Starting run_phase", UVM_LOW)
      run_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_run_phase(m_test_top, run_phase_h);

      // Wait for all run_phase tasks to complete via objection mechanism
      // Give a brief time for objections to be raised
      #1;
      wait_for_objections(run_phase_h);
      run_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed run_phase", UVM_LOW)

      // EXTRACT PHASE
      `uvm_info("UVM/PHASE", "Starting extract_phase", UVM_LOW)
      extract_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, extract_phase_h, "extract");
      extract_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed extract_phase", UVM_LOW)

      // CHECK PHASE
      `uvm_info("UVM/PHASE", "Starting check_phase", UVM_LOW)
      check_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, check_phase_h, "check");
      check_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed check_phase", UVM_LOW)

      // REPORT PHASE
      `uvm_info("UVM/PHASE", "Starting report_phase", UVM_LOW)
      report_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, report_phase_h, "report");
      report_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed report_phase", UVM_LOW)

      // FINAL PHASE
      `uvm_info("UVM/PHASE", "Starting final_phase", UVM_LOW)
      final_phase_h.set_state(UVM_PHASE_EXECUTING);
      execute_phase_func(m_test_top, final_phase_h, "final");
      final_phase_h.set_state(UVM_PHASE_DONE);
      `uvm_info("UVM/PHASE", "Completed final_phase", UVM_LOW)

      `uvm_info("UVM/RUNTST", "*** UVM TEST PASSED ***", UVM_NONE)
    endtask

  endclass

  // Global access to uvm_root
  uvm_root uvm_top = uvm_root::get();

  //=========================================================================
  // uvm_agent - Base class for agents
  //=========================================================================
  class uvm_agent extends uvm_component;
    uvm_active_passive_enum is_active = UVM_ACTIVE;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_agent";
    endfunction

    virtual function uvm_active_passive_enum get_is_active();
      return is_active;
    endfunction

  endclass

  //=========================================================================
  // uvm_driver - Base class for drivers
  //=========================================================================
  class uvm_driver #(type REQ = uvm_sequence_item, type RSP = REQ) extends uvm_component;
    uvm_seq_item_pull_port #(REQ, RSP) seq_item_port;
    uvm_analysis_port #(RSP) rsp_port;
    REQ req;
    RSP rsp;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      seq_item_port = new("seq_item_port", this);
      rsp_port = new("rsp_port", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_driver";
    endfunction

  endclass

  //=========================================================================
  // uvm_monitor - Base class for monitors
  //=========================================================================
  class uvm_monitor extends uvm_component;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_monitor";
    endfunction

  endclass

  //=========================================================================
  // uvm_scoreboard - Base class for scoreboards
  //=========================================================================
  class uvm_scoreboard extends uvm_component;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_scoreboard";
    endfunction

  endclass

  //=========================================================================
  // uvm_subscriber - Base class for analysis components
  //=========================================================================
  class uvm_subscriber #(type T = uvm_sequence_item) extends uvm_component;
    uvm_analysis_imp #(T, uvm_subscriber #(T)) analysis_export;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      analysis_export = new("analysis_export", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_subscriber";
    endfunction

    // Pure virtual - must be implemented by derived class
    virtual function void write(T t);
    endfunction

  endclass

  //=========================================================================
  // uvm_env - Base class for environments
  //=========================================================================
  class uvm_env extends uvm_component;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_env";
    endfunction

  endclass

  //=========================================================================
  // uvm_test - Base class for tests
  //=========================================================================
  class uvm_test extends uvm_component;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_test";
    endfunction

  endclass

  //=========================================================================
  // uvm_sequence_base - Base class for sequences (non-parameterized)
  //=========================================================================
  class uvm_sequence_base extends uvm_sequence_item;
    protected uvm_sequencer_base m_sequencer;
    protected uvm_sequence_base m_parent_sequence;
    protected bit m_sequence_state;
    protected int unsigned m_priority = 100;
    protected event m_sequence_started;

    function new(string name = "uvm_sequence");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequence_base";
    endfunction

    virtual task body();
      // Override in derived class
    endtask

    virtual task start(uvm_sequencer_base sequencer,
                       uvm_sequence_base parent_sequence = null,
                       int this_priority = -1,
                       bit call_pre_post = 1);
      m_sequencer = sequencer;
      m_parent_sequence = parent_sequence;
      if (this_priority >= 0)
        m_priority = this_priority;

      -> m_sequence_started;

      if (call_pre_post)
        pre_body();

      body();

      if (call_pre_post)
        post_body();
    endtask

    virtual task pre_body();
    endtask

    virtual task post_body();
    endtask

    virtual task pre_start();
    endtask

    virtual task post_start();
    endtask

    virtual function uvm_sequencer_base get_sequencer();
      return m_sequencer;
    endfunction

    virtual function void m_set_p_sequencer();
      // Override in derived class
    endfunction

    virtual function bit is_item();
      return 0;
    endfunction

    // Wait for sequencer to grant permission to send item
    virtual task wait_for_grant(int item_priority = -1, bit lock_request = 0);
      if (m_sequencer != null) begin
        m_sequencer.wait_for_grant(this, item_priority, lock_request);
      end
    endtask

    // Send request item to sequencer
    virtual function void send_request(uvm_sequence_item request, bit rerandomize = 0);
      if (m_sequencer != null) begin
        m_sequencer.send_request(this, request, rerandomize);
      end
    endfunction

    // Wait for driver to signal item processing is done
    virtual task wait_for_item_done(int transaction_id = -1);
      // Wait a delta cycle to allow driver to process
      // In real UVM this would wait for item_done() call
      #0;
    endtask

    // Item execution methods - the main API for sequences
    // start_item: Request permission from sequencer, set up item context
    virtual task start_item(uvm_sequence_item item, int set_priority = -1,
                            uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      if (sequencer == null) begin
        `uvm_error("SEQ", "start_item called with null sequencer")
        return;
      end
      item.set_item_context(this, sequencer);
      wait_for_grant(set_priority);
    endtask

    // finish_item: Send the item to driver and wait for completion
    virtual task finish_item(uvm_sequence_item item, int set_priority = -1);
      if (m_sequencer == null) begin
        `uvm_error("SEQ", "finish_item called with null sequencer")
        return;
      end
      send_request(item);
      wait_for_item_done(item.get_transaction_id());
    endtask

    virtual function int unsigned get_priority();
      return m_priority;
    endfunction

    virtual function void set_priority(int unsigned value);
      m_priority = value;
    endfunction

    // Check if this sequence is blocked by another sequence
    virtual function bit is_blocked();
      if (m_sequencer != null)
        return m_sequencer.is_blocked(this);
      return 0;
    endfunction

    // Check if this sequence holds the lock
    virtual function bit has_lock();
      if (m_sequencer != null)
        return m_sequencer.is_locked(this);
      return 0;
    endfunction

    // Lock the sequencer - waits for existing locks to clear
    virtual task lock(uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      if (sequencer != null)
        sequencer.lock(this);
    endtask

    // Unlock the sequencer
    virtual task unlock(uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      if (sequencer != null)
        sequencer.unlock(this);
    endtask

    // Grab the sequencer - immediate exclusive access (preempts lock)
    virtual task grab(uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      if (sequencer != null)
        sequencer.grab(this);
    endtask

    // Ungrab the sequencer
    virtual task ungrab(uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      if (sequencer != null)
        sequencer.ungrab(this);
    endtask

    // Response queue management
    protected bit m_use_response_handler = 0;
    protected int m_response_queue_depth = -1;
    protected bit m_response_queue_error_report_disabled = 0;

    virtual function void use_response_handler(bit enable);
      m_use_response_handler = enable;
    endfunction

    virtual function uvm_sequence_item get_base_response(int transaction_id = -1);
      return null;
    endfunction

    virtual function void set_response_queue_error_report_disabled(bit value);
      m_response_queue_error_report_disabled = value;
    endfunction

    virtual function bit get_response_queue_error_report_disabled();
      return m_response_queue_error_report_disabled;
    endfunction

    virtual function void set_response_queue_depth(int value);
      m_response_queue_depth = value;
    endfunction

    virtual function int get_response_queue_depth();
      return m_response_queue_depth;
    endfunction

  endclass

  //=========================================================================
  // uvm_sequence - Parameterized sequence class
  //=========================================================================
  class uvm_sequence #(type REQ = uvm_sequence_item, type RSP = REQ) extends uvm_sequence_base;
    REQ req;
    RSP rsp;

    protected RSP response_queue[$];
    protected event m_response_available;

    function new(string name = "uvm_sequence");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequence";
    endfunction

    // Get response from response queue (blocking)
    virtual task get_response(output RSP response, input int transaction_id = -1);
      while (response_queue.size() == 0) begin
        @(m_response_available);
      end
      if (transaction_id >= 0) begin
        // Find matching transaction_id
        foreach (response_queue[i]) begin
          if (response_queue[i].get_transaction_id() == transaction_id) begin
            response = response_queue[i];
            response_queue.delete(i);
            return;
          end
        end
        // Not found - wait for more responses
        response = null;
      end else begin
        // Return first response
        response = response_queue.pop_front();
      end
    endtask

    // Put response into queue
    virtual function void put_response(RSP response);
      // Check queue depth limit
      if (m_response_queue_depth > 0 && response_queue.size() >= m_response_queue_depth) begin
        if (!m_response_queue_error_report_disabled)
          `uvm_warning("RSP_OVERFLOW", "Response queue overflow - dropping oldest response")
        void'(response_queue.pop_front());
      end
      response_queue.push_back(response);
      -> m_response_available;
    endfunction

    // Check if response is available
    virtual function bit response_available();
      return (response_queue.size() > 0);
    endfunction

    // Get number of pending responses
    virtual function int num_responses();
      return response_queue.size();
    endfunction

  endclass

  //=========================================================================
  // uvm_sequencer_base - Base class for sequencers
  //=========================================================================
  class uvm_sequencer_base extends uvm_component;
    protected int m_arbitration;
    protected uvm_sequence_base m_sequences[$];
    protected int m_lock_arb_size;

    // Lock/grab state tracking
    protected uvm_sequence_base m_lock_list[$];
    protected uvm_sequence_base m_grab_list[$];
    protected event m_lock_changed;
    protected event m_item_available;
    protected int m_max_zero_time_wait_relevant_count = 10;

    // Grant tracking
    protected uvm_sequence_base m_grant_queue[$];
    protected bit m_grant_pending;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      m_arbitration = UVM_SEQ_ARB_FIFO;
      m_grant_pending = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_sequencer_base";
    endfunction

    virtual function void set_arbitration(uvm_sequencer_arb_mode mode);
      m_arbitration = mode;
    endfunction

    virtual function uvm_sequencer_arb_mode get_arbitration();
      return uvm_sequencer_arb_mode'(m_arbitration);
    endfunction

    // Check if sequence is blocked by another sequence with lock/grab
    virtual function int is_blocked(uvm_sequence_base seq);
      // If there's a grabbing sequence that isn't this one, we're blocked
      if (m_grab_list.size() > 0 && m_grab_list[0] != seq)
        return 1;
      // If there's a locked sequence that isn't this one or an ancestor, we're blocked
      foreach (m_lock_list[i]) begin
        if (m_lock_list[i] != seq)
          return 1;
      end
      return 0;
    endfunction

    // Check if this sequence currently holds the lock
    virtual function bit is_locked(uvm_sequence_base seq);
      foreach (m_lock_list[i]) begin
        if (m_lock_list[i] == seq)
          return 1;
      end
      foreach (m_grab_list[i]) begin
        if (m_grab_list[i] == seq)
          return 1;
      end
      return 0;
    endfunction

    // Wait for grant to send item - respects arbitration and lock/grab
    virtual task wait_for_grant(uvm_sequence_base seq, int item_priority = -1, bit lock_request = 0);
      // Add to grant queue
      m_grant_queue.push_back(seq);

      // Wait until we're at front and not blocked
      while (1) begin
        // Check if we can proceed
        if (m_grant_queue.size() > 0 && m_grant_queue[0] == seq && !is_blocked(seq)) begin
          m_grant_pending = 1;
          if (lock_request)
            m_lock_list.push_back(seq);
          return;
        end
        // Wait for state change
        @(m_lock_changed or m_item_available);
      end
    endtask

    // Wait for sequences to be available
    virtual task wait_for_sequences();
      if (m_sequences.size() == 0)
        @(m_item_available);
    endtask

    // User-overridable arbitration function
    virtual function int user_priority_arbitration(int avail_sequences[$]);
      if (avail_sequences.size() > 0)
        return avail_sequences[0];
      return -1;
    endfunction

    // Stop all running sequences
    virtual function void stop_sequences();
      m_sequences.delete();
      m_grant_queue.delete();
      m_lock_list.delete();
      m_grab_list.delete();
      -> m_lock_changed;
    endfunction

    // Lock sequencer for exclusive access - waits for any locks ahead
    virtual task lock(uvm_sequence_base seq);
      // Wait for existing locks to clear (cooperative locking)
      while (m_lock_list.size() > 0 || m_grab_list.size() > 0) begin
        @(m_lock_changed);
      end
      m_lock_list.push_back(seq);
      -> m_lock_changed;
    endtask

    // Release lock held by sequence
    virtual task unlock(uvm_sequence_base seq);
      foreach (m_lock_list[i]) begin
        if (m_lock_list[i] == seq) begin
          m_lock_list.delete(i);
          -> m_lock_changed;
          return;
        end
      end
    endtask

    // Grab sequencer - immediate exclusive access (preemptive)
    virtual task grab(uvm_sequence_base seq);
      // Grab takes priority - goes to front of grab list
      m_grab_list.push_front(seq);
      -> m_lock_changed;
    endtask

    // Release grab
    virtual task ungrab(uvm_sequence_base seq);
      foreach (m_grab_list[i]) begin
        if (m_grab_list[i] == seq) begin
          m_grab_list.delete(i);
          -> m_lock_changed;
          return;
        end
      end
    endtask

    virtual function void set_max_zero_time_wait_relevant_count(int value);
      m_max_zero_time_wait_relevant_count = value;
    endfunction

    // Signal that an item is available
    virtual function void m_signal_item_available();
      -> m_item_available;
    endfunction

    // Complete grant - remove from queue
    virtual function void m_complete_grant(uvm_sequence_base seq);
      if (m_grant_queue.size() > 0 && m_grant_queue[0] == seq) begin
        void'(m_grant_queue.pop_front());
        m_grant_pending = 0;
        -> m_lock_changed;
      end
    endfunction

    // Check if there are pending grants
    virtual function bit has_pending_grant();
      return m_grant_pending;
    endfunction

    // Virtual send_request stub - overridden in uvm_sequencer_param_base
    // This allows uvm_sequence_base to call send_request on m_sequencer
    // (which is declared as uvm_sequencer_base) without compile-time errors
    virtual function void send_request(uvm_sequence_base sequence_ptr, uvm_sequence_item t, bit rerandomize = 0);
      // Default implementation does nothing - subclass must override
      `uvm_warning("SQRBASE", "send_request called on uvm_sequencer_base - must use parameterized sequencer")
    endfunction

  endclass

  //=========================================================================
  // uvm_sequencer_param_base - Parameterized sequencer base
  //=========================================================================
  class uvm_sequencer_param_base #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_sequencer_base;

    uvm_seq_item_pull_imp #(REQ, RSP, uvm_sequencer_param_base #(REQ, RSP)) seq_item_export;

    protected REQ m_req_fifo[$];
    protected RSP m_rsp_fifo[$];
    protected event m_req_available;
    protected event m_rsp_available;
    protected bit m_item_in_progress;
    protected REQ m_current_item;
    protected uvm_sequence_base m_current_sequence;

    // Response queue for sequences
    protected RSP m_response_queue[$];
    protected int m_response_queue_depth = -1;  // -1 means unlimited
    protected bit m_response_queue_error_report_disabled = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      seq_item_export = new("seq_item_export", this);
      m_item_in_progress = 0;
      m_current_item = null;
      m_current_sequence = null;
    endfunction

    virtual function string get_type_name();
      return "uvm_sequencer_param_base";
    endfunction

    // Driver calls this to get next item (blocking)
    virtual task get_next_item(output REQ t);
      // Wait for item to be available
      while (m_req_fifo.size() == 0) begin
        @(m_req_available);
      end
      t = m_req_fifo[0];  // Don't remove yet - wait for item_done
      m_current_item = t;
      m_item_in_progress = 1;
    endtask

    // Driver calls this to try to get next item (non-blocking)
    virtual task try_next_item(output REQ t);
      if (m_req_fifo.size() > 0 && !m_item_in_progress) begin
        t = m_req_fifo[0];
        m_current_item = t;
        m_item_in_progress = 1;
      end else begin
        t = null;
      end
    endtask

    // Driver calls this when done processing item
    virtual function void item_done(RSP rsp = null);
      if (m_item_in_progress && m_req_fifo.size() > 0) begin
        void'(m_req_fifo.pop_front());
        m_item_in_progress = 0;
        m_current_item = null;

        // Handle response
        if (rsp != null) begin
          put_response(rsp);
        end

        // Signal completion to sequence
        if (m_current_sequence != null) begin
          m_complete_grant(m_current_sequence);
          m_current_sequence = null;
        end

        // Signal item available event for next item
        -> m_req_available;
      end
    endfunction

    // Put response into response queue
    virtual task put(RSP rsp);
      put_response(rsp);
    endtask

    // Get response from response queue (blocking)
    virtual task get(output RSP rsp);
      while (m_rsp_fifo.size() == 0) begin
        @(m_rsp_available);
      end
      rsp = m_rsp_fifo.pop_front();
    endtask

    // Check if there's an item available
    virtual function bit has_do_available();
      return (m_req_fifo.size() > 0) && !m_item_in_progress;
    endfunction

    // Peek at next item without removing
    virtual task peek(output REQ t);
      while (m_req_fifo.size() == 0) begin
        @(m_req_available);
      end
      t = m_req_fifo[0];
    endtask

    // Sequence calls this to send a request item
    virtual function void send_request(uvm_sequence_base sequence_ptr, uvm_sequence_item t, bit rerandomize = 0);
      REQ req;
      if ($cast(req, t)) begin
        if (rerandomize)
          void'(req.randomize());
        m_req_fifo.push_back(req);
        m_current_sequence = sequence_ptr;
        -> m_req_available;
        m_signal_item_available();
      end
    endfunction

    // Get response from base type
    virtual task get_base_response(output uvm_sequence_item rsp, input int transaction_id = -1);
      RSP typed_rsp;
      while (m_rsp_fifo.size() == 0) begin
        @(m_rsp_available);
      end
      // Find response with matching transaction_id if specified
      if (transaction_id >= 0) begin
        foreach (m_rsp_fifo[i]) begin
          if (m_rsp_fifo[i].get_transaction_id() == transaction_id) begin
            typed_rsp = m_rsp_fifo[i];
            m_rsp_fifo.delete(i);
            rsp = typed_rsp;
            return;
          end
        end
      end
      // Otherwise return first response
      typed_rsp = m_rsp_fifo.pop_front();
      rsp = typed_rsp;
    endtask

    // Put response into response queue
    virtual function void put_response(uvm_sequence_item rsp);
      RSP typed_rsp;
      if ($cast(typed_rsp, rsp)) begin
        // Check queue depth limit
        if (m_response_queue_depth > 0 && m_rsp_fifo.size() >= m_response_queue_depth) begin
          if (!m_response_queue_error_report_disabled)
            `uvm_warning("SEQ_RSP_OVERFLOW", "Response queue overflow - dropping oldest response")
          void'(m_rsp_fifo.pop_front());
        end
        m_rsp_fifo.push_back(typed_rsp);
        -> m_rsp_available;
      end
    endfunction

    // Get number of pending requests
    virtual function int get_num_reqs();
      return m_req_fifo.size();
    endfunction

    // Get number of pending responses
    virtual function int get_num_rsps();
      return m_rsp_fifo.size();
    endfunction

    // Set response queue depth
    virtual function void set_response_queue_depth(int depth);
      m_response_queue_depth = depth;
    endfunction

    // Get response queue depth
    virtual function int get_response_queue_depth();
      return m_response_queue_depth;
    endfunction

    // Set response queue error report disabled
    virtual function void set_response_queue_error_report_disabled(bit disabled);
      m_response_queue_error_report_disabled = disabled;
    endfunction

    // Get response queue error report disabled
    virtual function bit get_response_queue_error_report_disabled();
      return m_response_queue_error_report_disabled;
    endfunction

  endclass

  //=========================================================================
  // uvm_sequencer - Standard sequencer
  //=========================================================================
  class uvm_sequencer #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_sequencer_param_base #(REQ, RSP);
    typedef uvm_component_registry #(uvm_sequencer #(REQ, RSP),
                                     "uvm_sequencer") type_id;
    static function type_id get_type();
      return type_id::get();
    endfunction

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequencer";
    endfunction

  endclass

  //=========================================================================
  // uvm_virtual_sequencer - Convenience base for virtual sequencers
  //=========================================================================
  class uvm_virtual_sequencer extends uvm_sequencer #(uvm_sequence_item);
    typedef uvm_component_registry #(uvm_virtual_sequencer,
                                     "uvm_virtual_sequencer") type_id;
    static function type_id get_type();
      return type_id::get();
    endfunction

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_virtual_sequencer";
    endfunction

  endclass

  //=========================================================================
  // TLM Interface Classes
  //=========================================================================

  // TLM interface base (virtual)
  virtual class uvm_tlm_if_base #(type T1 = int, type T2 = int);
    // Blocking put
    virtual task put(input T1 t);
    endtask

    // Blocking get
    virtual task get(output T2 t);
    endtask

    // Blocking peek
    virtual task peek(output T2 t);
    endtask

    // Non-blocking try_put
    virtual function bit try_put(input T1 t);
      return 0;
    endfunction

    // Non-blocking can_put
    virtual function bit can_put();
      return 0;
    endfunction

    // Non-blocking try_get
    virtual function bit try_get(output T2 t);
      return 0;
    endfunction

    // Non-blocking can_get
    virtual function bit can_get();
      return 0;
    endfunction

    // Non-blocking try_peek
    virtual function bit try_peek(output T2 t);
      return 0;
    endfunction

    // Non-blocking can_peek
    virtual function bit can_peek();
      return 0;
    endfunction

    // Analysis write
    virtual function void write(input T1 t);
    endfunction

    // Transport
    virtual task transport(input T1 req, output T2 rsp);
    endtask

    virtual function bit nb_transport(input T1 req, output T2 rsp);
      return 0;
    endfunction

  endclass

  //=========================================================================
  // uvm_port_base - Base class for TLM ports
  //=========================================================================
  class uvm_port_base #(type IF = uvm_tlm_if_base #(int, int)) extends uvm_component;
    protected IF m_if;
    protected uvm_port_type_e m_port_type;
    protected int unsigned m_min_size;
    protected int unsigned m_max_size;
    protected uvm_port_base #(IF) m_provided_by[$];
    protected uvm_port_base #(IF) m_provided_to[$];

    function new(string name, uvm_component parent,
                 uvm_port_type_e port_type = UVM_PORT,
                 int min_size = 0, int max_size = 1);
      super.new(name, parent);
      m_port_type = port_type;
      m_min_size = min_size;
      m_max_size = max_size;
    endfunction

    virtual function string get_type_name();
      return "uvm_port_base";
    endfunction

    virtual function void connect(uvm_port_base #(IF) provider);
      m_provided_by.push_back(provider);
      provider.m_provided_to.push_back(this);
      m_if = provider.m_if;
    endfunction

    virtual function void set_if(IF if_inst);
      m_if = if_inst;
    endfunction

    virtual function IF get_if();
      return m_if;
    endfunction

    virtual function int size();
      return m_provided_by.size();
    endfunction

    virtual function int max_size();
      return m_max_size;
    endfunction

    virtual function int min_size();
      return m_min_size;
    endfunction

    virtual function bit is_unbounded();
      return m_max_size == 0;
    endfunction

    virtual function bit is_port();
      return m_port_type == UVM_PORT;
    endfunction

    virtual function bit is_export();
      return m_port_type == UVM_EXPORT;
    endfunction

    virtual function bit is_imp();
      return m_port_type == UVM_IMPLEMENTATION;
    endfunction

  endclass

  //=========================================================================
  // uvm_analysis_port - Analysis broadcast port
  //=========================================================================
  class uvm_analysis_port #(type T = int) extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    protected uvm_tlm_if_base #(T, T) m_subscribers[$];

    function new(string name, uvm_component parent);
      super.new(name, parent, UVM_PORT, 0, 0);
    endfunction

    virtual function string get_type_name();
      return "uvm_analysis_port";
    endfunction

    virtual function void connect(uvm_port_base #(uvm_tlm_if_base #(T, T)) provider);
      super.connect(provider);
      if (provider.m_if != null)
        m_subscribers.push_back(provider.m_if);
    endfunction

    virtual function void write(input T t);
      foreach (m_subscribers[i])
        m_subscribers[i].write(t);
    endfunction

  endclass

  //=========================================================================
  // uvm_analysis_imp - Analysis implementation port
  //=========================================================================
  class uvm_analysis_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_analysis_imp";
    endfunction

    virtual function void write(input T t);
      m_imp.write(t);
    endfunction

  endclass

  //=========================================================================
  // uvm_analysis_export - Analysis export
  //=========================================================================
  class uvm_analysis_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent);
      super.new(name, parent, UVM_EXPORT, 0, 1);
    endfunction

    virtual function string get_type_name();
      return "uvm_analysis_export";
    endfunction

    virtual function void write(input T t);
      if (m_if != null)
        m_if.write(t);
    endfunction

  endclass

  //=========================================================================
  // uvm_seq_item_pull_port - Sequencer to driver communication
  //=========================================================================
  // Note: For seq_item_pull ports, the TLM interface direction is reversed:
  // - put operations send RSP (responses to sequencer)
  // - get operations receive REQ (requests from sequencer)
  // So we parameterize as uvm_tlm_if_base #(RSP, REQ) where T1=RSP for put, T2=REQ for get
  class uvm_seq_item_pull_port #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_seq_item_pull_port";
    endfunction

    virtual task get_next_item(output REQ req_arg);
      if (m_if != null)
        m_if.get(req_arg);
    endtask

    virtual task try_next_item(output REQ req_arg);
      if (m_if != null && m_if.try_get(req_arg))
        ;
      else
        req_arg = null;
    endtask

    virtual function void item_done(RSP rsp = null);
      if (m_if != null && rsp != null)
        void'(m_if.try_put(rsp));
    endfunction

    virtual task put(RSP rsp);
      if (m_if != null)
        m_if.put(rsp);
    endtask

    virtual task get(output REQ req);
      if (m_if != null)
        m_if.get(req);
    endtask

    virtual task peek(output REQ req);
      if (m_if != null)
        m_if.peek(req);
    endtask

    virtual function void put_response(RSP rsp);
      if (m_if != null)
        void'(m_if.try_put(rsp));
    endfunction

    virtual function bit has_do_available();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

  endclass

  //=========================================================================
  // uvm_seq_item_pull_imp - Sequencer implementation
  //=========================================================================
  // Same reversed parameterization as uvm_seq_item_pull_port
  class uvm_seq_item_pull_imp #(type REQ = uvm_sequence_item, type RSP = REQ, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_seq_item_pull_imp";
    endfunction

    virtual task get(output REQ req);
      m_imp.get_next_item(req);
    endtask

    virtual function bit try_get(output REQ req);
      req = null;
      return 0;
    endfunction

    virtual task peek(output REQ req);
      m_imp.peek(req);
    endtask

    virtual function bit try_put(input RSP rsp);
      m_imp.item_done(rsp);
      return 1;
    endfunction

    virtual task put(input RSP rsp);
      m_imp.put(rsp);
    endtask

    virtual function bit can_get();
      return m_imp.has_do_available();
    endfunction

    virtual function bit can_peek();
      return m_imp.has_do_available();
    endfunction

  endclass

  //=========================================================================
  // uvm_seq_item_pull_export - Sequencer export
  //=========================================================================
  // Same reversed parameterization as uvm_seq_item_pull_port
  class uvm_seq_item_pull_export #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_seq_item_pull_export";
    endfunction

  endclass

  //=========================================================================
  // uvm_blocking_put_port - Blocking put port
  //=========================================================================
  class uvm_blocking_put_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 1, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task put(input T t);
      if (m_if != null)
        m_if.put(t);
    endtask

  endclass

  //=========================================================================
  // uvm_blocking_get_port - Blocking get port
  //=========================================================================
  class uvm_blocking_get_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 1, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

  endclass

  //=========================================================================
  // uvm_blocking_peek_port - Blocking peek port
  //=========================================================================
  class uvm_blocking_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 1, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

  endclass

  //=========================================================================
  // uvm_blocking_get_peek_port - Combined blocking get/peek port
  //=========================================================================
  class uvm_blocking_get_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 1, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

  endclass

  //=========================================================================
  // uvm_tlm_fifo - TLM FIFO
  //=========================================================================
  class uvm_tlm_fifo #(type T = int) extends uvm_component;
    protected T m_fifo[$];
    protected int unsigned m_max_size;

    // Exports
    uvm_put_imp #(T, uvm_tlm_fifo #(T)) put_export;
    uvm_get_peek_imp #(T, uvm_tlm_fifo #(T)) get_peek_export;
    uvm_analysis_port #(T) put_ap;
    uvm_analysis_port #(T) get_ap;

    function new(string name, uvm_component parent = null, int size = 1);
      super.new(name, parent);
      m_max_size = size;
      put_export = new("put_export", this);
      get_peek_export = new("get_peek_export", this);
      put_ap = new("put_ap", this);
      get_ap = new("get_ap", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_tlm_fifo";
    endfunction

    virtual function int size();
      return m_fifo.size();
    endfunction

    virtual function int used();
      return m_fifo.size();
    endfunction

    virtual function bit is_empty();
      return m_fifo.size() == 0;
    endfunction

    virtual function bit is_full();
      return m_max_size != 0 && m_fifo.size() >= m_max_size;
    endfunction

    virtual task put(input T t);
      m_fifo.push_back(t);
      put_ap.write(t);
    endtask

    virtual function bit try_put(input T t);
      if (is_full())
        return 0;
      m_fifo.push_back(t);
      put_ap.write(t);
      return 1;
    endfunction

    virtual function bit can_put();
      return !is_full();
    endfunction

    virtual task get(output T t);
      // Block until item is available (simplified - waits in a loop)
      wait (m_fifo.size() > 0);
      t = m_fifo.pop_front();
      get_ap.write(t);
    endtask

    virtual function bit try_get(output T t);
      if (is_empty()) begin
        // Don't assign null - T may be a non-class type
        return 0;
      end
      t = m_fifo.pop_front();
      get_ap.write(t);
      return 1;
    endfunction

    virtual function bit can_get();
      return !is_empty();
    endfunction

    virtual task peek(output T t);
      // Block until item is available
      wait (m_fifo.size() > 0);
      t = m_fifo[0];
    endtask

    virtual function bit try_peek(output T t);
      if (is_empty()) begin
        // Don't assign null - T may be a non-class type
        return 0;
      end
      t = m_fifo[0];
      return 1;
    endfunction

    virtual function bit can_peek();
      return !is_empty();
    endfunction

    virtual function void flush();
      m_fifo.delete();
    endfunction

  endclass

  //=========================================================================
  // uvm_tlm_analysis_fifo - Analysis FIFO
  //=========================================================================
  class uvm_tlm_analysis_fifo #(type T = int) extends uvm_tlm_fifo #(T);
    uvm_analysis_imp #(T, uvm_tlm_analysis_fifo #(T)) analysis_export;

    function new(string name, uvm_component parent = null);
      super.new(name, parent, 0); // Unbounded
      analysis_export = new("analysis_export", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_tlm_analysis_fifo";
    endfunction

    virtual function void write(input T t);
      void'(try_put(t));
    endfunction

  endclass

  //=========================================================================
  // Helper implementation classes
  //=========================================================================

  // uvm_put_imp
  class uvm_put_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual task put(input T t);
      m_imp.put(t);
    endtask

    virtual function bit try_put(input T t);
      return m_imp.try_put(t);
    endfunction

    virtual function bit can_put();
      return m_imp.can_put();
    endfunction

  endclass

  // uvm_get_imp
  class uvm_get_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual task get(output T t);
      m_imp.get(t);
    endtask

    virtual function bit try_get(output T t);
      return m_imp.try_get(t);
    endfunction

    virtual function bit can_get();
      return m_imp.can_get();
    endfunction

  endclass

  // uvm_get_peek_imp
  class uvm_get_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual task get(output T t);
      m_imp.get(t);
    endtask

    virtual function bit try_get(output T t);
      return m_imp.try_get(t);
    endfunction

    virtual function bit can_get();
      return m_imp.can_get();
    endfunction

    virtual task peek(output T t);
      m_imp.peek(t);
    endtask

    virtual function bit try_peek(output T t);
      return m_imp.try_peek(t);
    endfunction

    virtual function bit can_peek();
      return m_imp.can_peek();
    endfunction

  endclass

  //=========================================================================
  // uvm_config_db - Configuration database with wildcard support
  //=========================================================================
  // This implementation supports:
  //   - Hierarchical path construction from component context
  //   - Wildcard patterns (* and ?) in inst_name during set()
  //   - Wildcard pattern matching during get()
  //   - Proper key construction per IEEE 1800.2-2017
  //=========================================================================
  class uvm_config_db #(type T = int);

    // Storage for exact values
    static local T m_db[string];

    // Storage for pattern-based values (wildcards)
    // Key is the full pattern path, field_name is stored separately
    typedef struct {
      string inst_pattern;
      string field_name;
      T value;
    } config_entry_t;
    static local config_entry_t m_pattern_db[$];

    // Construct the lookup path from context and inst_name
    static function string m_get_lookup_path(uvm_component cntxt, string inst_name);
      string cntxt_name;
      if (cntxt == null)
        return inst_name;
      cntxt_name = cntxt.get_full_name();
      if (cntxt_name == "")
        return inst_name;
      if (inst_name == "")
        return cntxt_name;
      return {cntxt_name, ".", inst_name};
    endfunction

    // Check if a string contains wildcard characters
    static function bit m_has_wildcards(string s);
      for (int i = 0; i < s.len(); i++)
        if (s[i] == "*" || s[i] == "?")
          return 1;
      return 0;
    endfunction

    // Set a configuration value
    // IEEE 1800.2-2017: inst_name supports wildcards (* and ?)
    static function void set(uvm_component cntxt, string inst_name,
                             string field_name, T value);
      string inst_path;
      string key;

      // Construct the instance path
      inst_path = m_get_lookup_path(cntxt, inst_name);

      // Check if this is a wildcard pattern
      if (m_has_wildcards(inst_path)) begin
        // Store in pattern database
        config_entry_t entry;
        entry.inst_pattern = inst_path;
        entry.field_name = field_name;
        entry.value = value;
        // Remove any existing entry with same pattern and field
        foreach (m_pattern_db[i]) begin
          if (m_pattern_db[i].inst_pattern == inst_path &&
              m_pattern_db[i].field_name == field_name) begin
            m_pattern_db.delete(i);
            break;
          end
        end
        m_pattern_db.push_back(entry);
      end
      else begin
        // Store in exact-match database
        key = {inst_path, ".", field_name};
        m_db[key] = value;
      end
    endfunction

    // Get a configuration value
    // Lookup order:
    //   1. Exact match on full path
    //   2. Wildcard pattern matches (last matching pattern wins)
    //   3. Fallback lookups for compatibility
    static function bit get(uvm_component cntxt, string inst_name,
                            string field_name, ref T value);
      string lookup_path;
      string key;
      int match_idx;

      lookup_path = m_get_lookup_path(cntxt, inst_name);

      // 1. Try exact match
      key = {lookup_path, ".", field_name};
      if (m_db.exists(key)) begin
        value = m_db[key];
        return 1;
      end

      // 2. Try wildcard patterns (last match wins for same precedence)
      match_idx = -1;
      foreach (m_pattern_db[i]) begin
        if (m_pattern_db[i].field_name == field_name &&
            uvm_is_match(m_pattern_db[i].inst_pattern, lookup_path)) begin
          match_idx = i;
        end
      end
      if (match_idx >= 0) begin
        value = m_pattern_db[match_idx].value;
        return 1;
      end

      // 3. Fallback: try without context prefix
      if (cntxt != null && inst_name != "") begin
        key = {inst_name, ".", field_name};
        if (m_db.exists(key)) begin
          value = m_db[key];
          return 1;
        end
        // Check patterns against inst_name alone
        foreach (m_pattern_db[i]) begin
          if (m_pattern_db[i].field_name == field_name &&
              uvm_is_match(m_pattern_db[i].inst_pattern, inst_name)) begin
            value = m_pattern_db[i].value;
            return 1;
          end
        end
      end

      // 4. Fallback: try just field name (for "*" patterns)
      foreach (m_pattern_db[i]) begin
        if (m_pattern_db[i].field_name == field_name &&
            m_pattern_db[i].inst_pattern == "*") begin
          value = m_pattern_db[i].value;
          return 1;
        end
      end

      return 0;
    endfunction

    // Check if a configuration value exists
    static function bit exists(uvm_component cntxt, string inst_name,
                               string field_name, bit spell_chk = 0);
      string lookup_path;
      string key;

      lookup_path = m_get_lookup_path(cntxt, inst_name);

      // Check exact match
      key = {lookup_path, ".", field_name};
      if (m_db.exists(key))
        return 1;

      // Check wildcard patterns
      foreach (m_pattern_db[i]) begin
        if (m_pattern_db[i].field_name == field_name &&
            uvm_is_match(m_pattern_db[i].inst_pattern, lookup_path))
          return 1;
      end

      return 0;
    endfunction

    // Wait for a configuration value to be modified
    // Note: This is a stub - full implementation requires event-based notification
    static task wait_modified(uvm_component cntxt, string inst_name, string field_name);
      // In a full implementation, this would:
      // 1. Register a waiter for the given path/field
      // 2. Block until set() is called for a matching path/field
      // For now, just return immediately (non-blocking stub)
      #0; // Zero-delay to make it a valid task
    endtask

  endclass

  //=========================================================================
  // uvm_resource_db - Resource database (simpler than config_db)
  //=========================================================================
  class uvm_resource_db #(type T = int);
    static local T m_db[string];

    static function void set(string scope, string name, T val, uvm_object accessor = null);
      m_db[{scope, ".", name}] = val;
    endfunction

    static function bit read_by_name(string scope, string name, ref T val, input uvm_object accessor = null);
      string key = {scope, ".", name};
      if (m_db.exists(key)) begin
        val = m_db[key];
        return 1;
      end
      return 0;
    endfunction

    static function bit read_by_type(string scope, ref T val, input uvm_object accessor = null);
      return 0; // Stub
    endfunction

    static function void set_default(string scope, string name);
    endfunction

  endclass

  //=========================================================================
  // Factory Support Classes
  //=========================================================================

  // Factory wrapper base
  class uvm_object_wrapper extends uvm_void;
    virtual function uvm_object create_object(string name = "");
      return null;
    endfunction

    virtual function uvm_component create_component(string name, uvm_component parent);
      return null;
    endfunction

    virtual function string get_type_name();
      return "";
    endfunction
  endclass

  // Factory
  class uvm_factory;
    static local uvm_factory m_inst;
    protected uvm_object_wrapper m_type_names[string];
    protected uvm_object_wrapper m_types[string];
    protected string m_type_overrides[string];
    protected string m_inst_overrides[string];

    function new();
    endfunction

    static function uvm_factory get();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

    virtual function void register(uvm_object_wrapper obj);
      m_type_names[obj.get_type_name()] = obj;
    endfunction

    virtual function uvm_object create_object_by_name(string requested_type_name,
                                                      string parent_inst_path = "",
                                                      string name = "");
      string override_type = requested_type_name;
      string full_inst_path;
      string inst_key;

      // Build the full instance path for instance override lookup
      if (parent_inst_path != "" && name != "")
        full_inst_path = {parent_inst_path, ".", name};
      else if (parent_inst_path != "")
        full_inst_path = parent_inst_path;
      else
        full_inst_path = name;

      // Check instance overrides first (higher priority)
      inst_key = {full_inst_path, ":", requested_type_name};
      if (m_inst_overrides.exists(inst_key))
        override_type = m_inst_overrides[inst_key];
      // Then check type overrides
      else if (m_type_overrides.exists(requested_type_name))
        override_type = m_type_overrides[requested_type_name];

      if (m_type_names.exists(override_type))
        return m_type_names[override_type].create_object(name);
      return null;
    endfunction

    virtual function uvm_component create_component_by_name(string requested_type_name,
                                                            string parent_inst_path = "",
                                                            string name = "",
                                                            uvm_component parent = null);
      string override_type = requested_type_name;
      string full_inst_path;
      string inst_key;

      // Build the full instance path for instance override lookup
      if (parent_inst_path != "" && name != "")
        full_inst_path = {parent_inst_path, ".", name};
      else if (parent_inst_path != "")
        full_inst_path = parent_inst_path;
      else
        full_inst_path = name;

      // Check instance overrides first (higher priority)
      inst_key = {full_inst_path, ":", requested_type_name};
      if (m_inst_overrides.exists(inst_key))
        override_type = m_inst_overrides[inst_key];
      // Then check type overrides
      else if (m_type_overrides.exists(requested_type_name))
        override_type = m_type_overrides[requested_type_name];

      if (m_type_names.exists(override_type))
        return m_type_names[override_type].create_component(name, parent);
      return null;
    endfunction

    virtual function void set_type_override_by_name(string original_type_name,
                                                    string override_type_name,
                                                    bit replace = 1);
      if (replace || !m_type_overrides.exists(original_type_name))
        m_type_overrides[original_type_name] = override_type_name;
    endfunction

    virtual function void set_inst_override_by_name(string original_type_name,
                                                    string override_type_name,
                                                    string full_inst_path);
      m_inst_overrides[{full_inst_path, ":", original_type_name}] = override_type_name;
    endfunction

    virtual function void set_type_override_by_type(uvm_object_wrapper original_type,
                                                    uvm_object_wrapper override_type,
                                                    bit replace = 1);
      set_type_override_by_name(original_type.get_type_name(),
                                override_type.get_type_name(),
                                replace);
    endfunction

    virtual function void set_inst_override_by_type(uvm_object_wrapper original_type,
                                                    uvm_object_wrapper override_type,
                                                    string full_inst_path);
      set_inst_override_by_name(original_type.get_type_name(),
                                override_type.get_type_name(),
                                full_inst_path);
    endfunction

    virtual function void print(int all_types = 1);
      $display("UVM Factory:");
      foreach (m_type_names[name])
        $display("  %s", name);
    endfunction

  endclass

  // Global factory instance
  uvm_factory factory = uvm_factory::get();

  // Component registry - for `uvm_component_utils
  class uvm_component_registry #(type T = uvm_component, string Tname = "<unknown>")
    extends uvm_object_wrapper;

    local static uvm_component_registry #(T, Tname) me;
    local static T m_inst;

    static function uvm_component_registry #(T, Tname) get();
      if (me == null) begin
        me = new();
        factory.register(me);
      end
      return me;
    endfunction

    virtual function string get_type_name();
      return Tname;
    endfunction

    virtual function uvm_component create_component(string name, uvm_component parent);
      T obj;
      obj = new(name, parent);
      return obj;
    endfunction

    static function T create(string name, uvm_component parent, string contxt = "");
      uvm_object obj;
      uvm_factory f = uvm_factory::get();
      obj = f.create_component_by_name(Tname, contxt, name, parent);
      if (!$cast(m_inst, obj))
        return null;
      return m_inst;
    endfunction

    static function void set_type_override(uvm_object_wrapper override_type, bit replace = 1);
      factory.set_type_override_by_name(Tname, override_type.get_type_name(), replace);
    endfunction

    static function void set_inst_override(uvm_object_wrapper override_type,
                                           string inst_path,
                                           uvm_component parent = null);
      string full_path;
      if (parent != null)
        full_path = {parent.get_full_name(), ".", inst_path};
      else
        full_path = inst_path;
      factory.set_inst_override_by_name(Tname, override_type.get_type_name(), full_path);
    endfunction

  endclass

  // Object registry - for `uvm_object_utils
  class uvm_object_registry #(type T = uvm_object, string Tname = "<unknown>")
    extends uvm_object_wrapper;

    local static uvm_object_registry #(T, Tname) me;
    local static T m_inst;

    static function uvm_object_registry #(T, Tname) get();
      if (me == null) begin
        me = new();
        factory.register(me);
      end
      return me;
    endfunction

    virtual function string get_type_name();
      return Tname;
    endfunction

    virtual function uvm_object create_object(string name = "");
      T obj;
      obj = new(name);
      return obj;
    endfunction

    static function T create(string name = "", uvm_component parent = null, string contxt = "");
      uvm_object obj;
      uvm_factory f = uvm_factory::get();
      obj = f.create_object_by_name(Tname, contxt, name);
      if (!$cast(m_inst, obj))
        return null;
      return m_inst;
    endfunction

    static function void set_type_override(uvm_object_wrapper override_type, bit replace = 1);
      factory.set_type_override_by_name(Tname, override_type.get_type_name(), replace);
    endfunction

  endclass

  //=========================================================================
  // Utility Classes
  //=========================================================================

  // uvm_printer - Printing support
  class uvm_printer extends uvm_object;
    int knobs_depth = -1;
    bit knobs_reference = 1;
    uvm_radix_enum knobs_default_radix = UVM_HEX;
    string knobs_type_name = "";

    function new(string name = "printer");
      super.new(name);
    endfunction

    virtual function void print_int(string name, uvm_bitstream_t value, int size,
                                    uvm_radix_enum radix = UVM_NORADIX,
                                    byte scope_separator = ".",
                                    string type_name = "");
      $display("  %s: %0d", name, value);
    endfunction

    virtual function void print_field(string name, uvm_bitstream_t value,
                                      int size,
                                      uvm_radix_enum radix = UVM_NORADIX);
      print_int(name, value, size, radix);
    endfunction

    virtual function void print_string(string name, string value,
                                       byte scope_separator = ".");
      $display("  %s: %s", name, value);
    endfunction

    virtual function void print_object(string name, uvm_object value,
                                       byte scope_separator = ".");
      if (value != null)
        $display("  %s: %s", name, value.sprint());
    endfunction

    virtual function void print_array_header(string name, int size,
                                             string arraytype = "array",
                                             byte scope_separator = ".");
      $display("  %s[%0d]:", name, size);
    endfunction

    virtual function void print_array_footer(int size = 0);
    endfunction

  endclass

  // Convenience printer types
  class uvm_table_printer extends uvm_printer;
    function new(string name = "table_printer");
      super.new(name);
    endfunction
  endclass

  class uvm_tree_printer extends uvm_printer;
    function new(string name = "tree_printer");
      super.new(name);
    endfunction
  endclass

  class uvm_line_printer extends uvm_printer;
    function new(string name = "line_printer");
      super.new(name);
    endfunction
  endclass

  // uvm_comparer - Comparison support
  class uvm_comparer extends uvm_object;
    int unsigned show_max = 1;
    int unsigned verbosity = UVM_LOW;
    uvm_severity sev = UVM_INFO;
    bit physical = 1;
    bit abstract_ = 1;
    bit check_type = 1;
    int unsigned result = 0;
    int unsigned miscompares = 0;

    function new(string name = "comparer");
      super.new(name);
    endfunction

    virtual function bit compare_field(string name, uvm_bitstream_t lhs, uvm_bitstream_t rhs, int size,
                                       uvm_radix_enum radix = UVM_NORADIX);
      if (lhs !== rhs) begin
        miscompares++;
        return 0;
      end
      return 1;
    endfunction

    virtual function bit compare_field_int(string name, uvm_integral_t lhs, uvm_integral_t rhs, int size,
                                           uvm_radix_enum radix = UVM_NORADIX);
      if (lhs !== rhs) begin
        miscompares++;
        return 0;
      end
      return 1;
    endfunction

    virtual function bit compare_string(string name, string lhs, string rhs);
      if (lhs != rhs) begin
        miscompares++;
        return 0;
      end
      return 1;
    endfunction

    virtual function bit compare_object(string name, uvm_object lhs, uvm_object rhs);
      if (lhs == null && rhs == null)
        return 1;
      if (lhs == null || rhs == null) begin
        miscompares++;
        return 0;
      end
      return lhs.compare(rhs, this);
    endfunction

  endclass

  // uvm_packer - Pack/unpack support
  class uvm_packer extends uvm_object;
    bit big_endian = 1;
    bit physical = 1;
    bit abstract_ = 1;
    bit use_metadata = 0;
    protected bit m_bits[$];

    function new(string name = "packer");
      super.new(name);
    endfunction

    virtual function void pack_field(uvm_bitstream_t value, int size);
      for (int i = size - 1; i >= 0; i--)
        m_bits.push_back(value[i]);
    endfunction

    virtual function void pack_field_int(uvm_integral_t value, int size);
      for (int i = size - 1; i >= 0; i--)
        m_bits.push_back(value[i]);
    endfunction

    virtual function void pack_string(string value);
      for (int i = 0; i < value.len(); i++)
        pack_field_int(value[i], 8);
      pack_field_int(0, 8); // Null terminator
    endfunction

    virtual function void pack_object(uvm_object value);
      if (value != null)
        value.pack(m_bits, this);
    endfunction

    virtual function uvm_bitstream_t unpack_field(int size);
      uvm_bitstream_t value = 0;
      for (int i = size - 1; i >= 0 && m_bits.size() > 0; i--)
        value[i] = m_bits.pop_front();
      return value;
    endfunction

    virtual function uvm_integral_t unpack_field_int(int size);
      return unpack_field(size);
    endfunction

    virtual function string unpack_string();
      string value = "";
      byte c;
      while (m_bits.size() > 0) begin
        c = unpack_field_int(8);
        if (c == 0)
          break;
        value = {value, string'(c)};
      end
      return value;
    endfunction

    virtual function void unpack_object(uvm_object value);
      if (value != null)
        void'(value.unpack(m_bits, this));
    endfunction

    virtual function void reset();
      m_bits.delete();
    endfunction

    virtual function int get_packed_size();
      return m_bits.size();
    endfunction

  endclass

  // uvm_recorder - Transaction recording
  class uvm_recorder extends uvm_object;
    function new(string name = "recorder");
      super.new(name);
    endfunction

    virtual function void record_field(string name, uvm_bitstream_t value, int size,
                                       uvm_radix_enum radix = UVM_NORADIX);
    endfunction

    virtual function void record_field_int(string name, uvm_integral_t value, int size,
                                           uvm_radix_enum radix = UVM_NORADIX);
    endfunction

    virtual function void record_string(string name, string value);
    endfunction

    virtual function void record_object(string name, uvm_object value);
    endfunction

    virtual function void record_time(string name, time value);
    endfunction

    virtual function void record_generic(string name, string value, string type_name = "");
    endfunction

  endclass

  //=========================================================================
  // Callback Support (needed before RAL classes)
  //=========================================================================
  class uvm_callback extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction
  endclass

  class uvm_callbacks #(type T = uvm_object, type CB = uvm_callback);
    // Internal storage: associative array mapping object to queue of callbacks
    // Using string key based on object's unique ID for portability
    static CB m_cb_pool[string][$];
    // Type-wide (global) callbacks that apply to all objects of type T
    static CB m_type_cbs[$];

    // Get unique key for an object (null key for type-wide callbacks)
    static function string get_obj_key(T obj);
      if (obj == null)
        return "__null__";
      else
        return $sformatf("%0d", obj.get_inst_id());
    endfunction

    // Add a callback for a specific object or type-wide (if obj is null)
    static function void add(T obj, CB cb, uvm_apprepend ordering = UVM_APPEND);
      string key;
      if (cb == null) return;

      if (obj == null) begin
        // Type-wide callback
        if (ordering == UVM_APPEND)
          m_type_cbs.push_back(cb);
        else
          m_type_cbs.push_front(cb);
      end else begin
        // Object-specific callback
        key = get_obj_key(obj);
        if (!m_cb_pool.exists(key)) begin
          CB empty_queue[$];
          m_cb_pool[key] = empty_queue;
        end
        if (ordering == UVM_APPEND)
          m_cb_pool[key].push_back(cb);
        else
          m_cb_pool[key].push_front(cb);
      end
    endfunction

    // Delete a callback from a specific object or type-wide (if obj is null)
    static function void delete(T obj, CB cb);
      string key;
      int idx;
      if (cb == null) return;

      if (obj == null) begin
        // Delete from type-wide callbacks
        foreach (m_type_cbs[i]) begin
          if (m_type_cbs[i] == cb) begin
            m_type_cbs.delete(i);
            return;
          end
        end
      end else begin
        // Delete from object-specific callbacks
        key = get_obj_key(obj);
        if (m_cb_pool.exists(key)) begin
          foreach (m_cb_pool[key][i]) begin
            if (m_cb_pool[key][i] == cb) begin
              m_cb_pool[key].delete(i);
              return;
            end
          end
        end
      end
    endfunction

    // Get all callbacks for an object (includes type-wide + object-specific)
    static function void get(T obj, ref CB cbs[$]);
      string key;
      cbs.delete();  // Clear the output queue

      // First add type-wide callbacks
      foreach (m_type_cbs[i]) begin
        cbs.push_back(m_type_cbs[i]);
      end

      // Then add object-specific callbacks
      if (obj != null) begin
        key = get_obj_key(obj);
        if (m_cb_pool.exists(key)) begin
          foreach (m_cb_pool[key][i]) begin
            cbs.push_back(m_cb_pool[key][i]);
          end
        end
      end
    endfunction

    // Get number of callbacks for an object
    static function int get_count(T obj);
      string key;
      int count = m_type_cbs.size();
      if (obj != null) begin
        key = get_obj_key(obj);
        if (m_cb_pool.exists(key))
          count += m_cb_pool[key].size();
      end
      return count;
    endfunction

    // Check if a callback is registered for an object
    static function bit is_registered(T obj, CB cb);
      string key;
      if (cb == null) return 0;

      // Check type-wide callbacks
      foreach (m_type_cbs[i]) begin
        if (m_type_cbs[i] == cb) return 1;
      end

      // Check object-specific callbacks
      if (obj != null) begin
        key = get_obj_key(obj);
        if (m_cb_pool.exists(key)) begin
          foreach (m_cb_pool[key][i]) begin
            if (m_cb_pool[key][i] == cb) return 1;
          end
        end
      end
      return 0;
    endfunction

    // Display registered callbacks (for debugging)
    static function void display(T obj = null);
      string key;
      $display("=== uvm_callbacks display ===");
      $display("Type-wide callbacks: %0d", m_type_cbs.size());
      foreach (m_type_cbs[i]) begin
        $display("  [%0d] %s", i, m_type_cbs[i].get_name());
      end
      if (obj != null) begin
        key = get_obj_key(obj);
        if (m_cb_pool.exists(key)) begin
          $display("Object-specific callbacks for %s: %0d", obj.get_name(), m_cb_pool[key].size());
          foreach (m_cb_pool[key][i]) begin
            $display("  [%0d] %s", i, m_cb_pool[key][i].get_name());
          end
        end else begin
          $display("No object-specific callbacks for %s", obj.get_name());
        end
      end
      $display("=============================");
    endfunction
  endclass

  //=========================================================================
  // uvm_callback_iter - Iterator for traversing callbacks
  //=========================================================================
  class uvm_callback_iter #(type T = uvm_object, type CB = uvm_callback);
    local T m_obj;
    local CB m_cbs[$];
    local int m_idx;

    function new(T obj);
      m_obj = obj;
      m_idx = -1;
      uvm_callbacks#(T, CB)::get(obj, m_cbs);
    endfunction

    // Get first callback, returns null if no callbacks
    function CB first();
      m_idx = 0;
      if (m_cbs.size() > 0)
        return m_cbs[0];
      return null;
    endfunction

    // Get last callback, returns null if no callbacks
    function CB last();
      if (m_cbs.size() > 0) begin
        m_idx = m_cbs.size() - 1;
        return m_cbs[m_idx];
      end
      return null;
    endfunction

    // Get next callback, returns null if at end
    function CB next();
      m_idx++;
      if (m_idx < m_cbs.size())
        return m_cbs[m_idx];
      return null;
    endfunction

    // Get previous callback, returns null if at beginning
    function CB prev();
      m_idx--;
      if (m_idx >= 0)
        return m_cbs[m_idx];
      return null;
    endfunction

    // Get callback at current index
    function CB get_cb();
      if (m_idx >= 0 && m_idx < m_cbs.size())
        return m_cbs[m_idx];
      return null;
    endfunction
  endclass

  //=========================================================================
  // RAL (Register Abstraction Layer) - IEEE 1800.2 Compliant Stubs
  //=========================================================================

  // Additional RAL enumerations
  typedef enum {
    UVM_NO_CHECK,
    UVM_CHECK
  } uvm_check_e;

  typedef enum {
    UVM_NO_ENDIAN,
    UVM_LITTLE_ENDIAN,
    UVM_BIG_ENDIAN,
    UVM_LITTLE_FIFO,
    UVM_BIG_FIFO
  } uvm_endianness_e;

  typedef enum {
    UVM_PREDICT_DIRECT,
    UVM_PREDICT_READ,
    UVM_PREDICT_WRITE
  } uvm_predict_e;

  typedef enum {
    UVM_COVERAGE_MODEL_ON  = 'h01,
    UVM_COVERAGE_FIELD_VALS = 'h02,
    UVM_COVERAGE_ADDR_MAP  = 'h04,
    UVM_CVR_ALL            = 'hFF,
    UVM_NO_COVERAGE        = 'h00
  } uvm_coverage_model_e;

  typedef enum {
    UVM_REG_NO_HIER,
    UVM_REG_HIER
  } uvm_hier_e;

  typedef enum {
    UVM_REG_DATA_FRONT,
    UVM_REG_DATA_BACK
  } uvm_reg_mem_tests_e;

  // Forward declarations for RAL classes
  typedef class uvm_reg_block;
  typedef class uvm_reg_file;
  typedef class uvm_reg_frontdoor;
  typedef class uvm_reg_backdoor;
  typedef class uvm_reg_cbs;
  typedef class uvm_reg_predictor;
  typedef class uvm_reg_item;
  typedef class uvm_hdl_path_slice;
  typedef class uvm_hdl_path_concat;

  // Typedef for coverage type (must be declared before use)
  typedef int uvm_reg_cvr_t;

  //=========================================================================
  // uvm_reg_adapter - Register bus adapter base class
  //=========================================================================
  class uvm_reg_adapter extends uvm_object;
    // Configuration properties
    bit supports_byte_enable = 0;
    bit provides_responses = 0;
    bit parent_sequence_lock = 0;

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_adapter";
    endfunction

    // Convert register operation to bus transaction
    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      return null; // Override in derived class
    endfunction

    // Convert bus transaction to register operation
    virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
      // Override in derived class
    endfunction

    // Get parent sequence for hierarchical sequences
    virtual function uvm_sequence_base get_parent_sequence();
      return null;
    endfunction

    // Get item being adapted
    virtual function uvm_reg_item get_item();
      return null;
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_item - Register transaction item
  //=========================================================================
  class uvm_reg_item extends uvm_sequence_item;
    uvm_access_e kind;
    uvm_reg_data_t value[$];
    uvm_reg_addr_t offset;
    uvm_status_e status;
    uvm_reg_map map;
    uvm_reg_map local_map;
    uvm_reg element;
    uvm_object element_kind;
    uvm_path_e path;
    uvm_object extension;
    string bd_kind;
    uvm_reg_field field;
    uvm_reg_frontdoor frontdoor;
    uvm_sequence_base parent;
    int prior = -1;
    string fname;
    int lineno;

    function new(string name = "uvm_reg_item");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_item";
    endfunction

    virtual function string convert2string();
      string s;
      s = $sformatf("kind=%s path=%s status=%s",
                    kind.name(), path.name(), status.name());
      if (value.size() > 0)
        s = {s, $sformatf(" value[0]='h%0h", value[0])};
      return s;
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_field - Individual register field
  //=========================================================================
  class uvm_reg_field extends uvm_object;
    // Internal state
    protected uvm_reg m_parent;
    protected int unsigned m_size;
    protected int unsigned m_lsb_pos;
    protected string m_access;
    protected bit m_volatile;
    protected uvm_reg_data_t m_reset;
    protected bit m_has_reset;
    protected bit m_is_rand;
    protected bit m_individually_accessible;
    protected uvm_reg_data_t m_value;
    protected uvm_reg_data_t m_mirrored;
    protected uvm_reg_data_t m_desired;
    protected bit m_read_in_progress;
    protected bit m_write_in_progress;

    function new(string name = "");
      super.new(name);
      m_size = 0;
      m_lsb_pos = 0;
      m_access = "RW";
      m_volatile = 0;
      m_reset = 0;
      m_has_reset = 0;
      m_is_rand = 0;
      m_value = 0;
      m_mirrored = 0;
      m_desired = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_field";
    endfunction

    // Configuration
    virtual function void configure(uvm_reg parent,
                                    int unsigned size,
                                    int unsigned lsb_pos,
                                    string access,
                                    bit volatile_,
                                    uvm_reg_data_t reset,
                                    bit has_reset,
                                    bit is_rand,
                                    bit individually_accessible);
      m_parent = parent;
      m_size = size;
      m_lsb_pos = lsb_pos;
      m_access = access;
      m_volatile = volatile_;
      m_reset = reset;
      m_has_reset = has_reset;
      m_is_rand = is_rand;
      m_individually_accessible = individually_accessible;
      if (has_reset) begin
        m_value = reset;
        m_mirrored = reset;
        m_desired = reset;
      end
    endfunction

    // Introspection
    virtual function uvm_reg get_parent();
      return m_parent;
    endfunction

    virtual function string get_full_name();
      if (m_parent != null)
        return {m_parent.get_full_name(), ".", get_name()};
      return get_name();
    endfunction

    virtual function int unsigned get_n_bits();
      return m_size;
    endfunction

    virtual function int unsigned get_lsb_pos();
      return m_lsb_pos;
    endfunction

    virtual function string get_access(uvm_reg_map map = null);
      return m_access;
    endfunction

    virtual function bit is_volatile();
      return m_volatile;
    endfunction

    virtual function bit is_known_access(uvm_reg_map map = null);
      string access = get_access(map);
      return (access == "RO" || access == "RW" || access == "RC" ||
              access == "RS" || access == "WRC" || access == "WRS" ||
              access == "WC" || access == "WS" || access == "W1C" ||
              access == "W1S" || access == "W1T" || access == "W0C" ||
              access == "W0S" || access == "W0T" || access == "W1" ||
              access == "WO" || access == "WO1");
    endfunction

    // Value operations
    virtual function void set(uvm_reg_data_t value, string fname = "", int lineno = 0);
      uvm_reg_data_t mask = (1 << m_size) - 1;
      m_desired = value & mask;
    endfunction

    virtual function uvm_reg_data_t get(string fname = "", int lineno = 0);
      return m_desired;
    endfunction

    virtual function void set_mirrored_value(uvm_reg_data_t value);
      uvm_reg_data_t mask = (1 << m_size) - 1;
      m_mirrored = value & mask;
    endfunction

    virtual function uvm_reg_data_t get_mirrored_value(string fname = "", int lineno = 0);
      return m_mirrored;
    endfunction

    // Reset
    virtual function uvm_reg_data_t get_reset(string kind = "HARD");
      return m_reset;
    endfunction

    virtual function bit has_reset(string kind = "HARD", bit delete_ = 0);
      return m_has_reset;
    endfunction

    virtual function void set_reset(uvm_reg_data_t value, string kind = "HARD");
      m_reset = value;
      m_has_reset = 1;
    endfunction

    virtual function void reset(string kind = "HARD");
      if (m_has_reset) begin
        m_value = m_reset;
        m_mirrored = m_reset;
        m_desired = m_reset;
      end
    endfunction

    // Compare
    virtual function bit needs_update();
      return m_mirrored !== m_desired;
    endfunction

    // Prediction
    virtual function bit predict(uvm_reg_data_t value,
                                 uvm_reg_byte_en_t be = -1,
                                 uvm_predict_e kind = UVM_PREDICT_DIRECT,
                                 uvm_path_e path = UVM_FRONTDOOR,
                                 uvm_reg_map map = null,
                                 string fname = "", int lineno = 0);
      uvm_reg_data_t mask = (1 << m_size) - 1;
      m_mirrored = value & mask;
      return 1;
    endfunction

    // Access operations
    virtual task write(output uvm_status_e status,
                       input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      uvm_reg_data_t mask = (1 << m_size) - 1;
      m_write_in_progress = 1;
      m_value = value & mask;
      m_mirrored = m_value;
      m_desired = m_value;
      status = UVM_IS_OK;
      m_write_in_progress = 0;
    endtask

    virtual task read(output uvm_status_e status,
                      output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      m_read_in_progress = 1;
      value = m_mirrored;
      status = UVM_IS_OK;
      m_read_in_progress = 0;
    endtask

    virtual task poke(output uvm_status_e status,
                      input uvm_reg_data_t value,
                      input string kind = "",
                      input uvm_sequence_base parent = null,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      uvm_reg_data_t mask = (1 << m_size) - 1;
      m_value = value & mask;
      m_mirrored = m_value;
      status = UVM_IS_OK;
    endtask

    virtual task peek(output uvm_status_e status,
                      output uvm_reg_data_t value,
                      input string kind = "",
                      input uvm_sequence_base parent = null,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      value = m_value;
      status = UVM_IS_OK;
    endtask

    virtual task mirror(output uvm_status_e status,
                        input uvm_check_e check = UVM_NO_CHECK,
                        input uvm_path_e path = UVM_DEFAULT_PATH,
                        input uvm_reg_map map = null,
                        input uvm_sequence_base parent = null,
                        input int prior = -1,
                        input uvm_object extension = null,
                        input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
    endtask

    // Callback hooks
    virtual task pre_write(uvm_reg_item rw);
    endtask

    virtual task post_write(uvm_reg_item rw);
    endtask

    virtual task pre_read(uvm_reg_item rw);
    endtask

    virtual task post_read(uvm_reg_item rw);
    endtask

  endclass

  //=========================================================================
  // uvm_reg - Register containing fields
  //=========================================================================
  class uvm_reg extends uvm_object;
    // Internal state
    protected uvm_reg_block m_parent;
    protected uvm_reg_file m_regfile_parent;
    protected int unsigned m_n_bits;
    protected int m_has_cover;
    protected bit m_is_busy;
    protected bit m_locked;
    protected uvm_reg_field m_fields[$];
    protected uvm_reg_data_t m_reset;
    protected uvm_reg_frontdoor m_frontdoor;
    protected uvm_reg_backdoor m_backdoor;
    protected string m_hdl_paths_pool[string][$];
    protected uvm_reg_addr_t m_offset;
    protected bit m_maps[uvm_reg_map];

    function new(string name = "", int unsigned n_bits = 32, int has_coverage = 0);
      super.new(name);
      m_n_bits = n_bits;
      m_has_cover = has_coverage;
      m_is_busy = 0;
      m_locked = 0;
      m_reset = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_reg";
    endfunction

    // Configuration
    virtual function void configure(uvm_reg_block blk_parent,
                                    uvm_reg_file regfile_parent = null,
                                    string hdl_path = "");
      m_parent = blk_parent;
      m_regfile_parent = regfile_parent;
      if (hdl_path != "")
        add_hdl_path_slice(hdl_path, -1, -1);
    endfunction

    // Field management
    virtual function void add_field(uvm_reg_field field);
      m_fields.push_back(field);
    endfunction

    virtual function void get_fields(ref uvm_reg_field fields[$]);
      fields = m_fields;
    endfunction

    virtual function uvm_reg_field get_field_by_name(string name);
      foreach (m_fields[i])
        if (m_fields[i].get_name() == name)
          return m_fields[i];
      return null;
    endfunction

    virtual function int get_n_fields();
      return m_fields.size();
    endfunction

    // Introspection
    virtual function uvm_reg_block get_parent();
      return m_parent;
    endfunction

    virtual function uvm_reg_block get_block();
      return m_parent;
    endfunction

    virtual function uvm_reg_file get_regfile();
      return m_regfile_parent;
    endfunction

    virtual function string get_full_name();
      if (m_parent != null)
        return {m_parent.get_full_name(), ".", get_name()};
      return get_name();
    endfunction

    virtual function int unsigned get_n_bits();
      return m_n_bits;
    endfunction

    virtual function int unsigned get_n_bytes();
      return (m_n_bits + 7) / 8;
    endfunction

    virtual function uvm_reg_addr_t get_address(uvm_reg_map map = null);
      return m_offset;
    endfunction

    virtual function void get_addresses(uvm_reg_map map = null, ref uvm_reg_addr_t addr[]);
      addr = new[1];
      addr[0] = m_offset;
    endfunction

    virtual function uvm_reg_addr_t get_offset(uvm_reg_map map = null);
      return m_offset;
    endfunction

    virtual function void set_offset(uvm_reg_map map, uvm_reg_addr_t offset,
                                     bit unmapped = 0);
      m_offset = offset;
      m_maps[map] = 1;
    endfunction

    virtual function void get_maps(ref uvm_reg_map maps[$]);
      foreach (m_maps[m])
        maps.push_back(m);
    endfunction

    virtual function bit is_in_map(uvm_reg_map map);
      return m_maps.exists(map);
    endfunction

    virtual function uvm_reg_map get_default_map();
      uvm_reg_map map;
      if (m_maps.first(map))
        return map;
      return null;
    endfunction

    virtual function string get_access(uvm_reg_map map = null);
      return "RW"; // Simplified - would aggregate field accesses
    endfunction

    virtual function string get_rights(uvm_reg_map map = null);
      return "RW";
    endfunction

    // Value operations
    virtual function void set(uvm_reg_data_t value, string fname = "", int lineno = 0);
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        int size = m_fields[i].get_n_bits();
        uvm_reg_data_t field_val = (value >> lsb) & ((1 << size) - 1);
        m_fields[i].set(field_val);
      end
    endfunction

    virtual function uvm_reg_data_t get(string fname = "", int lineno = 0);
      uvm_reg_data_t value = 0;
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        value |= m_fields[i].get() << lsb;
      end
      return value;
    endfunction

    virtual function uvm_reg_data_t get_mirrored_value(string fname = "", int lineno = 0);
      uvm_reg_data_t value = 0;
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        value |= m_fields[i].get_mirrored_value() << lsb;
      end
      return value;
    endfunction

    // Reset
    virtual function uvm_reg_data_t get_reset(string kind = "HARD");
      uvm_reg_data_t value = 0;
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        value |= m_fields[i].get_reset(kind) << lsb;
      end
      return value;
    endfunction

    virtual function bit has_reset(string kind = "HARD", bit delete_ = 0);
      foreach (m_fields[i])
        if (m_fields[i].has_reset(kind))
          return 1;
      return 0;
    endfunction

    virtual function void set_reset(uvm_reg_data_t value, string kind = "HARD");
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        int size = m_fields[i].get_n_bits();
        uvm_reg_data_t field_val = (value >> lsb) & ((1 << size) - 1);
        m_fields[i].set_reset(field_val, kind);
      end
    endfunction

    virtual function void reset(string kind = "HARD");
      foreach (m_fields[i])
        m_fields[i].reset(kind);
    endfunction

    // Comparison
    virtual function bit needs_update();
      foreach (m_fields[i])
        if (m_fields[i].needs_update())
          return 1;
      return 0;
    endfunction

    // Prediction
    virtual function bit predict(uvm_reg_data_t value,
                                 uvm_reg_byte_en_t be = -1,
                                 uvm_predict_e kind = UVM_PREDICT_DIRECT,
                                 uvm_path_e path = UVM_FRONTDOOR,
                                 uvm_reg_map map = null,
                                 string fname = "", int lineno = 0);
      foreach (m_fields[i]) begin
        int lsb = m_fields[i].get_lsb_pos();
        int size = m_fields[i].get_n_bits();
        uvm_reg_data_t field_val = (value >> lsb) & ((1 << size) - 1);
        if (!m_fields[i].predict(field_val, be, kind, path, map, fname, lineno))
          return 0;
      end
      return 1;
    endfunction

    // Access operations
    virtual task write(output uvm_status_e status,
                       input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      uvm_reg_item rw = new("write_item");
      rw.kind = UVM_WRITE;
      rw.value.push_back(value);
      rw.path = path;
      rw.map = map;
      rw.parent = parent;
      rw.prior = prior;
      rw.extension = extension;
      rw.fname = fname;
      rw.lineno = lineno;

      pre_write(rw);
      // Simplified stub - just update internal state
      set(value);
      void'(predict(value, -1, UVM_PREDICT_WRITE, path, map));
      rw.status = UVM_IS_OK;
      post_write(rw);

      status = rw.status;
    endtask

    virtual task read(output uvm_status_e status,
                      output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      uvm_reg_item rw = new("read_item");
      rw.kind = UVM_READ;
      rw.path = path;
      rw.map = map;
      rw.parent = parent;
      rw.prior = prior;
      rw.extension = extension;
      rw.fname = fname;
      rw.lineno = lineno;

      pre_read(rw);
      // Simplified stub - just return mirrored value
      value = get_mirrored_value();
      rw.value.push_back(value);
      rw.status = UVM_IS_OK;
      void'(predict(value, -1, UVM_PREDICT_READ, path, map));
      post_read(rw);

      status = rw.status;
    endtask

    virtual task poke(output uvm_status_e status,
                      input uvm_reg_data_t value,
                      input string kind = "",
                      input uvm_sequence_base parent = null,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      foreach (m_fields[i]) begin
        uvm_status_e field_status;
        int lsb = m_fields[i].get_lsb_pos();
        int size = m_fields[i].get_n_bits();
        uvm_reg_data_t field_val = (value >> lsb) & ((1 << size) - 1);
        m_fields[i].poke(field_status, field_val, kind, parent, extension, fname, lineno);
        if (field_status != UVM_IS_OK)
          status = field_status;
      end
      status = UVM_IS_OK;
    endtask

    virtual task peek(output uvm_status_e status,
                      output uvm_reg_data_t value,
                      input string kind = "",
                      input uvm_sequence_base parent = null,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      value = 0;
      foreach (m_fields[i]) begin
        uvm_status_e field_status;
        uvm_reg_data_t field_val;
        int lsb = m_fields[i].get_lsb_pos();
        m_fields[i].peek(field_status, field_val, kind, parent, extension, fname, lineno);
        value |= field_val << lsb;
      end
      status = UVM_IS_OK;
    endtask

    virtual task update(output uvm_status_e status,
                        input uvm_path_e path = UVM_DEFAULT_PATH,
                        input uvm_reg_map map = null,
                        input uvm_sequence_base parent = null,
                        input int prior = -1,
                        input uvm_object extension = null,
                        input string fname = "", input int lineno = 0);
      if (needs_update())
        write(status, get(), path, map, parent, prior, extension, fname, lineno);
      else
        status = UVM_IS_OK;
    endtask

    virtual task mirror(output uvm_status_e status,
                        input uvm_check_e check = UVM_NO_CHECK,
                        input uvm_path_e path = UVM_DEFAULT_PATH,
                        input uvm_reg_map map = null,
                        input uvm_sequence_base parent = null,
                        input int prior = -1,
                        input uvm_object extension = null,
                        input string fname = "", input int lineno = 0);
      uvm_reg_data_t v;
      read(status, v, path, map, parent, prior, extension, fname, lineno);
    endtask

    // HDL path support
    virtual function void add_hdl_path(uvm_hdl_path_slice slices[], string kind = "RTL");
    endfunction

    virtual function void add_hdl_path_slice(string name, int offset, int size,
                                             bit first = 0, string kind = "RTL");
      if (!m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind] = {};
      m_hdl_paths_pool[kind].push_back(name);
    endfunction

    virtual function bit has_hdl_path(string kind = "");
      if (kind == "")
        return m_hdl_paths_pool.size() > 0;
      return m_hdl_paths_pool.exists(kind);
    endfunction

    virtual function void get_hdl_path(ref uvm_hdl_path_concat paths[$], input string kind = "");
    endfunction

    virtual function void get_hdl_path_kinds(ref string kinds[$]);
      foreach (m_hdl_paths_pool[k])
        kinds.push_back(k);
    endfunction

    virtual function string get_full_hdl_path(string kind = "", string separator = ".");
      string path = "";
      if (m_parent != null)
        path = m_parent.get_full_hdl_path(kind, separator);
      if (m_hdl_paths_pool.exists(kind) && m_hdl_paths_pool[kind].size() > 0) begin
        if (path != "")
          path = {path, separator, m_hdl_paths_pool[kind][0]};
        else
          path = m_hdl_paths_pool[kind][0];
      end
      return path;
    endfunction

    // Frontdoor/backdoor access
    virtual function void set_frontdoor(uvm_reg_frontdoor ftdr, uvm_reg_map map = null,
                                        string fname = "", int lineno = 0);
      m_frontdoor = ftdr;
    endfunction

    virtual function uvm_reg_frontdoor get_frontdoor(uvm_reg_map map = null);
      return m_frontdoor;
    endfunction

    virtual function void set_backdoor(uvm_reg_backdoor bkdr, string fname = "", int lineno = 0);
      m_backdoor = bkdr;
    endfunction

    virtual function uvm_reg_backdoor get_backdoor(bit inherit = 1);
      return m_backdoor;
    endfunction

    virtual function void clear_hdl_path(string kind = "RTL");
      if (m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind].delete();
    endfunction

    // Locking
    virtual function void lock_model();
      m_locked = 1;
    endfunction

    virtual function bit is_locked();
      return m_locked;
    endfunction

    // Coverage sampling
    virtual function void sample_values();
      // Override in derived class to implement coverage sampling
    endfunction

    virtual function void sample(uvm_reg_addr_t offset, bit is_read,
                                 uvm_reg_map map);
      // Override in derived class to implement coverage sampling
    endfunction

    // Callback hooks
    virtual task pre_write(uvm_reg_item rw);
    endtask

    virtual task post_write(uvm_reg_item rw);
    endtask

    virtual task pre_read(uvm_reg_item rw);
    endtask

    virtual task post_read(uvm_reg_item rw);
    endtask

  endclass

  //=========================================================================
  // uvm_reg_file - Register file container
  //=========================================================================
  class uvm_reg_file extends uvm_object;
    protected uvm_reg_block m_parent;
    protected uvm_reg_file m_rf_parent;
    protected string m_hdl_paths_pool[string][$];

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_file";
    endfunction

    virtual function void configure(uvm_reg_block blk_parent, uvm_reg_file regfile_parent,
                                    string hdl_path = "");
      m_parent = blk_parent;
      m_rf_parent = regfile_parent;
      if (hdl_path != "")
        add_hdl_path(hdl_path);
    endfunction

    virtual function string get_full_name();
      if (m_parent != null)
        return {m_parent.get_full_name(), ".", get_name()};
      return get_name();
    endfunction

    virtual function uvm_reg_block get_parent();
      return m_parent;
    endfunction

    virtual function uvm_reg_block get_block();
      return m_parent;
    endfunction

    virtual function uvm_reg_file get_regfile();
      return m_rf_parent;
    endfunction

    virtual function void add_hdl_path(string path, string kind = "RTL");
      if (!m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind] = {};
      m_hdl_paths_pool[kind].push_back(path);
    endfunction

    virtual function bit has_hdl_path(string kind = "");
      if (kind == "")
        return m_hdl_paths_pool.size() > 0;
      return m_hdl_paths_pool.exists(kind);
    endfunction

    virtual function void clear_hdl_path(string kind = "RTL");
      if (m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind].delete();
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_block - Block containing registers and memories
  //=========================================================================
  class uvm_reg_block extends uvm_object;
    // Internal state
    protected uvm_reg_block m_parent;
    protected int m_has_cover;
    protected bit m_locked;
    protected uvm_reg m_regs[$];
    protected uvm_mem m_mems[$];
    protected uvm_reg_block m_children[$];
    protected uvm_reg_map m_maps[$];
    protected uvm_reg_map m_default_map;
    protected uvm_reg_backdoor m_backdoor;
    protected string m_hdl_paths_pool[string][$];
    protected uvm_reg_file m_regfiles[$];

    function new(string name = "", int has_coverage = 0);
      super.new(name);
      m_has_cover = has_coverage;
      m_locked = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_block";
    endfunction

    // Configuration
    virtual function void configure(uvm_reg_block parent = null, string hdl_path = "");
      m_parent = parent;
      if (hdl_path != "")
        add_hdl_path(hdl_path);
      if (parent != null)
        parent.add_block(this);
    endfunction

    // Hierarchy
    virtual function uvm_reg_block get_parent();
      return m_parent;
    endfunction

    virtual function string get_full_name();
      if (m_parent != null)
        return {m_parent.get_full_name(), ".", get_name()};
      return get_name();
    endfunction

    virtual function void add_block(uvm_reg_block blk);
      m_children.push_back(blk);
    endfunction

    virtual function void get_blocks(ref uvm_reg_block blks[$], input uvm_hier_e hier = UVM_REG_HIER);
      blks = m_children;
      if (hier == UVM_REG_HIER)
        foreach (m_children[i])
          m_children[i].get_blocks(blks, hier);
    endfunction

    virtual function uvm_reg_block get_block_by_name(string name);
      foreach (m_children[i])
        if (m_children[i].get_name() == name)
          return m_children[i];
      return null;
    endfunction

    // Register management
    virtual function void add_reg(uvm_reg rg);
      m_regs.push_back(rg);
    endfunction

    virtual function void get_registers(ref uvm_reg regs[$], input uvm_hier_e hier = UVM_REG_HIER);
      regs = m_regs;
      if (hier == UVM_REG_HIER)
        foreach (m_children[i])
          m_children[i].get_registers(regs, hier);
    endfunction

    virtual function uvm_reg get_reg_by_name(string name);
      foreach (m_regs[i])
        if (m_regs[i].get_name() == name)
          return m_regs[i];
      return null;
    endfunction

    virtual function int get_n_registers(uvm_hier_e hier = UVM_REG_HIER);
      int n = m_regs.size();
      if (hier == UVM_REG_HIER)
        foreach (m_children[i])
          n += m_children[i].get_n_registers(hier);
      return n;
    endfunction

    // Memory management
    virtual function void add_mem(uvm_mem mem);
      m_mems.push_back(mem);
    endfunction

    virtual function void get_memories(ref uvm_mem mems[$], input uvm_hier_e hier = UVM_REG_HIER);
      mems = m_mems;
      if (hier == UVM_REG_HIER)
        foreach (m_children[i])
          m_children[i].get_memories(mems, hier);
    endfunction

    virtual function uvm_mem get_mem_by_name(string name);
      foreach (m_mems[i])
        if (m_mems[i].get_name() == name)
          return m_mems[i];
      return null;
    endfunction

    virtual function int get_n_memories(uvm_hier_e hier = UVM_REG_HIER);
      int n = m_mems.size();
      if (hier == UVM_REG_HIER)
        foreach (m_children[i])
          n += m_children[i].get_n_memories(hier);
      return n;
    endfunction

    // Register file management
    virtual function void add_regfile(uvm_reg_file regfile);
      m_regfiles.push_back(regfile);
    endfunction

    // Map management
    virtual function uvm_reg_map create_map(string name,
                                            uvm_reg_addr_t base_addr,
                                            int unsigned n_bytes,
                                            uvm_endianness_e endian,
                                            bit byte_addressing = 1);
      uvm_reg_map map = new(name);
      map.configure(this, base_addr, n_bytes, endian, byte_addressing);
      m_maps.push_back(map);
      if (m_default_map == null)
        m_default_map = map;
      return map;
    endfunction

    virtual function void get_maps(ref uvm_reg_map maps[$]);
      maps = m_maps;
    endfunction

    virtual function uvm_reg_map get_map_by_name(string name);
      foreach (m_maps[i])
        if (m_maps[i].get_name() == name)
          return m_maps[i];
      return null;
    endfunction

    virtual function uvm_reg_map get_default_map();
      return m_default_map;
    endfunction

    virtual function void set_default_map(uvm_reg_map map);
      m_default_map = map;
    endfunction

    virtual function int get_n_maps();
      return m_maps.size();
    endfunction

    // Locking
    virtual function void lock_model();
      m_locked = 1;
      foreach (m_regs[i])
        m_regs[i].lock_model();
      foreach (m_children[i])
        m_children[i].lock_model();
    endfunction

    virtual function bit is_locked();
      return m_locked;
    endfunction

    // Reset
    virtual function void reset(string kind = "HARD");
      foreach (m_regs[i])
        m_regs[i].reset(kind);
      foreach (m_children[i])
        m_children[i].reset(kind);
    endfunction

    virtual function bit needs_update();
      foreach (m_regs[i])
        if (m_regs[i].needs_update())
          return 1;
      foreach (m_children[i])
        if (m_children[i].needs_update())
          return 1;
      return 0;
    endfunction

    virtual task update(output uvm_status_e status,
                        input uvm_path_e path = UVM_DEFAULT_PATH,
                        input uvm_sequence_base parent = null,
                        input int prior = -1,
                        input uvm_object extension = null,
                        input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
      foreach (m_regs[i]) begin
        uvm_status_e s;
        m_regs[i].update(s, path, null, parent, prior, extension, fname, lineno);
        if (s != UVM_IS_OK)
          status = s;
      end
    endtask

    virtual task mirror(output uvm_status_e status,
                        input uvm_check_e check = UVM_NO_CHECK,
                        input uvm_path_e path = UVM_DEFAULT_PATH,
                        input uvm_sequence_base parent = null,
                        input int prior = -1,
                        input uvm_object extension = null,
                        input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
      foreach (m_regs[i]) begin
        uvm_status_e s;
        m_regs[i].mirror(s, check, path, null, parent, prior, extension, fname, lineno);
        if (s != UVM_IS_OK)
          status = s;
      end
    endtask

    // HDL path support
    virtual function void set_hdl_path_root(string path, string kind = "RTL");
      clear_hdl_path(kind);
      add_hdl_path(path, kind);
    endfunction

    virtual function void add_hdl_path(string path, string kind = "RTL");
      if (!m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind] = {};
      m_hdl_paths_pool[kind].push_back(path);
    endfunction

    virtual function bit has_hdl_path(string kind = "");
      if (kind == "")
        return m_hdl_paths_pool.size() > 0;
      return m_hdl_paths_pool.exists(kind);
    endfunction

    virtual function string get_hdl_path(string kind = "RTL");
      if (m_hdl_paths_pool.exists(kind) && m_hdl_paths_pool[kind].size() > 0)
        return m_hdl_paths_pool[kind][0];
      return "";
    endfunction

    virtual function string get_full_hdl_path(string kind = "RTL", string separator = ".");
      string path = "";
      if (m_parent != null)
        path = m_parent.get_full_hdl_path(kind, separator);
      if (m_hdl_paths_pool.exists(kind) && m_hdl_paths_pool[kind].size() > 0) begin
        if (path != "")
          path = {path, separator, m_hdl_paths_pool[kind][0]};
        else
          path = m_hdl_paths_pool[kind][0];
      end
      return path;
    endfunction

    virtual function void clear_hdl_path(string kind = "RTL");
      if (m_hdl_paths_pool.exists(kind))
        m_hdl_paths_pool[kind].delete();
    endfunction

    virtual function void get_hdl_path_kinds(ref string kinds[$]);
      foreach (m_hdl_paths_pool[k])
        kinds.push_back(k);
    endfunction

    // Backdoor access
    virtual function void set_backdoor(uvm_reg_backdoor bkdr, string fname = "", int lineno = 0);
      m_backdoor = bkdr;
    endfunction

    virtual function uvm_reg_backdoor get_backdoor(bit inherit = 1);
      if (m_backdoor != null)
        return m_backdoor;
      if (inherit && m_parent != null)
        return m_parent.get_backdoor();
      return null;
    endfunction

    // Coverage
    virtual function uvm_reg_cvr_t get_coverage(bit is_field = 0);
      return m_has_cover;
    endfunction

    virtual function bit has_coverage(uvm_reg_cvr_t models);
      return (m_has_cover & models) != 0;
    endfunction

    virtual function uvm_reg_cvr_t set_coverage(uvm_reg_cvr_t is_on);
      uvm_reg_cvr_t prev = m_has_cover;
      m_has_cover = is_on;
      return prev;
    endfunction

    virtual function void sample(uvm_reg_addr_t offset, bit is_read,
                                 uvm_reg_map map);
    endfunction

    virtual function void sample_values();
      foreach (m_regs[i])
        m_regs[i].sample_values();
      foreach (m_children[i])
        m_children[i].sample_values();
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_map - Address map
  //=========================================================================
  class uvm_reg_map extends uvm_object;
    // Internal state
    protected uvm_reg_block m_parent;
    protected uvm_reg_addr_t m_base_addr;
    protected int unsigned m_n_bytes;
    protected uvm_endianness_e m_endian;
    protected bit m_byte_addressing;
    protected uvm_sequencer_base m_sequencer;
    protected uvm_reg_adapter m_adapter;
    protected bit m_auto_predict;
    protected bit m_check_on_read;
    protected uvm_reg_map m_parent_map;
    protected uvm_reg_map m_submaps[$];
    protected uvm_reg m_regs_by_offset[uvm_reg_addr_t];
    protected uvm_mem m_mems_by_offset[uvm_reg_addr_t];
    protected uvm_reg_addr_t m_submap_offset[uvm_reg_map];

    function new(string name = "");
      super.new(name);
      m_base_addr = 0;
      m_n_bytes = 4;
      m_endian = UVM_LITTLE_ENDIAN;
      m_byte_addressing = 1;
      m_auto_predict = 0;
      m_check_on_read = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_map";
    endfunction

    // Configuration
    virtual function void configure(uvm_reg_block parent,
                                    uvm_reg_addr_t base_addr,
                                    int unsigned n_bytes,
                                    uvm_endianness_e endian,
                                    bit byte_addressing = 1);
      m_parent = parent;
      m_base_addr = base_addr;
      m_n_bytes = n_bytes;
      m_endian = endian;
      m_byte_addressing = byte_addressing;
    endfunction

    // Introspection
    virtual function uvm_reg_block get_parent();
      return m_parent;
    endfunction

    virtual function uvm_reg_block get_root_map();
      if (m_parent_map != null)
        return m_parent_map.get_root_map();
      return m_parent;
    endfunction

    virtual function string get_full_name();
      if (m_parent != null)
        return {m_parent.get_full_name(), ".", get_name()};
      return get_name();
    endfunction

    virtual function uvm_reg_addr_t get_base_addr(uvm_hier_e hier = UVM_REG_HIER);
      uvm_reg_addr_t addr = m_base_addr;
      if (hier == UVM_REG_HIER && m_parent_map != null)
        addr += m_parent_map.get_base_addr(hier);
      return addr;
    endfunction

    virtual function int unsigned get_n_bytes(uvm_hier_e hier = UVM_REG_HIER);
      if (hier == UVM_REG_HIER && m_parent_map != null)
        return m_parent_map.get_n_bytes(hier);
      return m_n_bytes;
    endfunction

    virtual function uvm_endianness_e get_endian(uvm_hier_e hier = UVM_REG_HIER);
      if (hier == UVM_REG_HIER && m_parent_map != null)
        return m_parent_map.get_endian(hier);
      return m_endian;
    endfunction

    virtual function bit get_addr_unit_bytes();
      return m_byte_addressing ? 1 : m_n_bytes;
    endfunction

    // Register/memory mapping
    virtual function void add_reg(uvm_reg rg, uvm_reg_addr_t offset,
                                  string rights = "RW", bit unmapped = 0,
                                  uvm_reg_frontdoor frontdoor = null);
      m_regs_by_offset[offset] = rg;
      rg.set_offset(this, offset, unmapped);
    endfunction

    virtual function void add_mem(uvm_mem mem, uvm_reg_addr_t offset,
                                  string rights = "RW", bit unmapped = 0,
                                  uvm_reg_frontdoor frontdoor = null);
      m_mems_by_offset[offset] = mem;
    endfunction

    virtual function uvm_reg get_reg_by_offset(uvm_reg_addr_t offset, bit read = 1);
      if (m_regs_by_offset.exists(offset))
        return m_regs_by_offset[offset];
      return null;
    endfunction

    virtual function uvm_mem get_mem_by_offset(uvm_reg_addr_t offset);
      if (m_mems_by_offset.exists(offset))
        return m_mems_by_offset[offset];
      return null;
    endfunction

    virtual function void get_registers(ref uvm_reg regs[$], input uvm_hier_e hier = UVM_REG_HIER);
      foreach (m_regs_by_offset[o])
        regs.push_back(m_regs_by_offset[o]);
    endfunction

    virtual function void get_memories(ref uvm_mem mems[$], input uvm_hier_e hier = UVM_REG_HIER);
      foreach (m_mems_by_offset[o])
        mems.push_back(m_mems_by_offset[o]);
    endfunction

    // Submap management
    virtual function void add_submap(uvm_reg_map child_map, uvm_reg_addr_t offset);
      m_submaps.push_back(child_map);
      m_submap_offset[child_map] = offset;
      child_map.m_parent_map = this;
    endfunction

    virtual function void get_submaps(ref uvm_reg_map submaps[$], input uvm_hier_e hier = UVM_REG_HIER);
      submaps = m_submaps;
    endfunction

    virtual function uvm_reg_map get_parent_map();
      return m_parent_map;
    endfunction

    virtual function uvm_reg_addr_t get_submap_offset(uvm_reg_map submap);
      if (m_submap_offset.exists(submap))
        return m_submap_offset[submap];
      return 0;
    endfunction

    // Sequencer/adapter binding
    virtual function void set_sequencer(uvm_sequencer_base sequencer,
                                        uvm_reg_adapter adapter = null);
      m_sequencer = sequencer;
      m_adapter = adapter;
    endfunction

    virtual function uvm_sequencer_base get_sequencer(uvm_hier_e hier = UVM_REG_HIER);
      if (m_sequencer != null)
        return m_sequencer;
      if (hier == UVM_REG_HIER && m_parent_map != null)
        return m_parent_map.get_sequencer(hier);
      return null;
    endfunction

    virtual function uvm_reg_adapter get_adapter(uvm_hier_e hier = UVM_REG_HIER);
      if (m_adapter != null)
        return m_adapter;
      if (hier == UVM_REG_HIER && m_parent_map != null)
        return m_parent_map.get_adapter(hier);
      return null;
    endfunction

    // Auto-predict
    virtual function void set_auto_predict(bit on = 1);
      m_auto_predict = on;
    endfunction

    virtual function bit get_auto_predict();
      return m_auto_predict;
    endfunction

    // Check on read
    virtual function void set_check_on_read(bit on = 1);
      m_check_on_read = on;
    endfunction

    virtual function bit get_check_on_read();
      return m_check_on_read;
    endfunction

    // Address calculation
    virtual function uvm_reg_addr_t get_addr_by_offset(uvm_reg_addr_t offset);
      return get_base_addr() + offset;
    endfunction

    virtual function void reset(string kind = "SOFT");
      // Reset internal state if needed
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_predictor - Prediction utility component
  //=========================================================================
  class uvm_reg_predictor #(type BUSTYPE = uvm_sequence_item) extends uvm_component;
    uvm_analysis_imp #(BUSTYPE, uvm_reg_predictor #(BUSTYPE)) bus_in;
    uvm_analysis_port #(uvm_reg_item) reg_ap;
    uvm_reg_map map;
    uvm_reg_adapter adapter;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      bus_in = new("bus_in", this);
      reg_ap = new("reg_ap", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_predictor";
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
    endfunction

    virtual function void write(BUSTYPE tr);
      uvm_reg rg;
      uvm_reg_bus_op rw;

      if (adapter == null) begin
        `uvm_error("NO_ADAPTER", "No adapter specified for predictor")
        return;
      end

      // Convert bus transaction to register operation
      adapter.bus2reg(tr, rw);

      if (map == null) begin
        `uvm_error("NO_MAP", "No map specified for predictor")
        return;
      end

      // Look up register at this address
      rg = map.get_reg_by_offset(rw.addr);
      if (rg != null) begin
        uvm_reg_item reg_item = new("predictor_item");
        reg_item.element = rg;
        reg_item.kind = rw.kind;
        reg_item.value.push_back(rw.data);
        reg_item.status = rw.status;

        // Predict based on operation type
        if (rw.kind == UVM_WRITE)
          void'(rg.predict(rw.data, -1, UVM_PREDICT_WRITE, UVM_FRONTDOOR, map));
        else
          void'(rg.predict(rw.data, -1, UVM_PREDICT_READ, UVM_FRONTDOOR, map));

        // Broadcast the register item
        reg_ap.write(reg_item);
      end
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_sequence - Register sequence base class
  //=========================================================================
  class uvm_reg_sequence #(type BASE = uvm_sequence #(uvm_reg_item)) extends BASE;
    uvm_reg_block model;
    uvm_reg_map reg_seqr;

    protected uvm_reg_adapter m_adapter;
    protected uvm_reg_map m_regs_maps[uvm_reg_map];

    function new(string name = "uvm_reg_sequence");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_sequence";
    endfunction

    // Write register using register model
    virtual task write_reg(input uvm_reg rg,
                           output uvm_status_e status,
                           input uvm_reg_data_t value,
                           input uvm_path_e path = UVM_DEFAULT_PATH,
                           input uvm_reg_map map = null,
                           input int prior = -1,
                           input uvm_object extension = null,
                           input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        return;
      end
      rg.write(status, value, path, map, this, prior, extension, fname, lineno);
    endtask

    // Read register using register model
    virtual task read_reg(input uvm_reg rg,
                          output uvm_status_e status,
                          output uvm_reg_data_t value,
                          input uvm_path_e path = UVM_DEFAULT_PATH,
                          input uvm_reg_map map = null,
                          input int prior = -1,
                          input uvm_object extension = null,
                          input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        value = 0;
        return;
      end
      rg.read(status, value, path, map, this, prior, extension, fname, lineno);
    endtask

    // Poke register using backdoor
    virtual task poke_reg(input uvm_reg rg,
                          output uvm_status_e status,
                          input uvm_reg_data_t value,
                          input string kind = "",
                          input uvm_object extension = null,
                          input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        return;
      end
      rg.poke(status, value, kind, this, extension, fname, lineno);
    endtask

    // Peek register using backdoor
    virtual task peek_reg(input uvm_reg rg,
                          output uvm_status_e status,
                          output uvm_reg_data_t value,
                          input string kind = "",
                          input uvm_object extension = null,
                          input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        value = 0;
        return;
      end
      rg.peek(status, value, kind, this, extension, fname, lineno);
    endtask

    // Update register (write if needed)
    virtual task update_reg(input uvm_reg rg,
                            output uvm_status_e status,
                            input uvm_path_e path = UVM_DEFAULT_PATH,
                            input uvm_reg_map map = null,
                            input int prior = -1,
                            input uvm_object extension = null,
                            input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        return;
      end
      rg.update(status, path, map, this, prior, extension, fname, lineno);
    endtask

    // Mirror register (read and optionally check)
    virtual task mirror_reg(input uvm_reg rg,
                            output uvm_status_e status,
                            input uvm_check_e check = UVM_NO_CHECK,
                            input uvm_path_e path = UVM_DEFAULT_PATH,
                            input uvm_reg_map map = null,
                            input int prior = -1,
                            input uvm_object extension = null,
                            input string fname = "", input int lineno = 0);
      if (rg == null) begin
        `uvm_error("NO_REG", "Register handle is null")
        status = UVM_NOT_OK;
        return;
      end
      rg.mirror(status, check, path, map, this, prior, extension, fname, lineno);
    endtask

    // Write memory
    virtual task write_mem(input uvm_mem mem,
                           output uvm_status_e status,
                           input uvm_reg_addr_t offset,
                           input uvm_reg_data_t value,
                           input uvm_path_e path = UVM_DEFAULT_PATH,
                           input uvm_reg_map map = null,
                           input int prior = -1,
                           input uvm_object extension = null,
                           input string fname = "", input int lineno = 0);
      if (mem == null) begin
        `uvm_error("NO_MEM", "Memory handle is null")
        status = UVM_NOT_OK;
        return;
      end
      mem.write(status, offset, value, path, map, this, prior, extension, fname, lineno);
    endtask

    // Read memory
    virtual task read_mem(input uvm_mem mem,
                          output uvm_status_e status,
                          input uvm_reg_addr_t offset,
                          output uvm_reg_data_t value,
                          input uvm_path_e path = UVM_DEFAULT_PATH,
                          input uvm_reg_map map = null,
                          input int prior = -1,
                          input uvm_object extension = null,
                          input string fname = "", input int lineno = 0);
      if (mem == null) begin
        `uvm_error("NO_MEM", "Memory handle is null")
        status = UVM_NOT_OK;
        value = 0;
        return;
      end
      mem.read(status, offset, value, path, map, this, prior, extension, fname, lineno);
    endtask

  endclass

  //=========================================================================
  // uvm_reg_frontdoor - Frontdoor access sequence
  //=========================================================================
  class uvm_reg_frontdoor extends uvm_reg_sequence #(uvm_sequence #(uvm_sequence_item));
    uvm_reg_item rw_info;
    uvm_sequencer_base sequencer;

    function new(string name = "uvm_reg_frontdoor");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_frontdoor";
    endfunction

    virtual task body();
      // Override in derived class to implement actual frontdoor access
    endtask

    virtual function bit is_active();
      return 1;
    endfunction

  endclass

  //=========================================================================
  // uvm_reg_backdoor - Backdoor access hook
  //=========================================================================
  class uvm_reg_backdoor extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_backdoor";
    endfunction

    // Write backdoor - override in derived class
    virtual task write(uvm_reg_item rw);
      `uvm_error("NOT_IMPL", "backdoor write() not implemented")
      rw.status = UVM_NOT_OK;
    endtask

    // Read backdoor - override in derived class
    virtual task read(uvm_reg_item rw);
      `uvm_error("NOT_IMPL", "backdoor read() not implemented")
      rw.status = UVM_NOT_OK;
    endtask

    // Functional read backdoor
    virtual function uvm_status_e read_func(uvm_reg_item rw);
      `uvm_error("NOT_IMPL", "backdoor read_func() not implemented")
      return UVM_NOT_OK;
    endfunction

    // Check if backdoor is available
    virtual function bit is_auto_updated(uvm_reg_field field);
      return 0;
    endfunction

    // Wait for update
    virtual local task wait_for_change(uvm_object element);
    endtask

    // Pre/post backdoor hooks
    virtual task pre_read(uvm_reg_item rw);
    endtask

    virtual task post_read(uvm_reg_item rw);
    endtask

    virtual task pre_write(uvm_reg_item rw);
    endtask

    virtual task post_write(uvm_reg_item rw);
    endtask

  endclass

  //=========================================================================
  // uvm_hdl_path_slice - HDL path slice descriptor
  //=========================================================================
  class uvm_hdl_path_slice;
    string path;
    int offset;
    int size;
  endclass

  //=========================================================================
  // uvm_hdl_path_concat - Concatenated HDL path
  //=========================================================================
  class uvm_hdl_path_concat;
    uvm_hdl_path_slice slices[$];

    function void set(uvm_hdl_path_slice slice);
      slices.push_back(slice);
    endfunction

    function void add_path(string path, int offset = -1, int size = -1);
      uvm_hdl_path_slice slice = new();
      slice.path = path;
      slice.offset = offset;
      slice.size = size;
      slices.push_back(slice);
    endfunction
  endclass

  //=========================================================================
  // uvm_reg_cbs - Register callback class
  //=========================================================================
  class uvm_reg_cbs extends uvm_callback;
    function new(string name = "uvm_reg_cbs");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_reg_cbs";
    endfunction

    // Pre/post register operation callbacks
    virtual task pre_write(uvm_reg_item rw);
    endtask

    virtual task post_write(uvm_reg_item rw);
    endtask

    virtual task pre_read(uvm_reg_item rw);
    endtask

    virtual task post_read(uvm_reg_item rw);
    endtask

    // Field-level callbacks
    virtual function void post_predict(input uvm_reg_field fld,
                                       input uvm_reg_data_t previous,
                                       inout uvm_reg_data_t value,
                                       input uvm_predict_e kind,
                                       input uvm_path_e path,
                                       input uvm_reg_map map);
    endfunction

    // Encode/decode callbacks
    virtual function void encode(ref uvm_reg_data_t data[$]);
    endfunction

    virtual function void decode(ref uvm_reg_data_t data[$]);
    endfunction

  endclass

  //=========================================================================
  // Event Support
  //=========================================================================
  class uvm_event #(type T = uvm_object) extends uvm_object;
    local event m_event;
    local T m_trigger_data;
    local bit m_is_on;
    local int m_num_waiters;

    function new(string name = "");
      super.new(name);
      m_is_on = 0;
      m_num_waiters = 0;
    endfunction

    virtual task wait_on(bit delta = 0);
    endtask

    virtual task wait_off(bit delta = 0);
    endtask

    virtual task wait_trigger();
    endtask

    virtual task wait_trigger_data(output T data);
      data = m_trigger_data;
    endtask

    virtual task wait_ptrigger();
    endtask

    virtual task wait_ptrigger_data(output T data);
      data = m_trigger_data;
    endtask

    virtual function void trigger(T data = null);
      m_is_on = 1;
      m_trigger_data = data;
    endfunction

    virtual function bit is_on();
      return m_is_on;
    endfunction

    virtual function bit is_off();
      return !m_is_on;
    endfunction

    virtual function void reset(bit wakeup = 0);
      m_is_on = 0;
      m_trigger_data = null;
    endfunction

    virtual function int get_num_waiters();
      return m_num_waiters;
    endfunction

    virtual function T get_trigger_data();
      return m_trigger_data;
    endfunction

  endclass

  class uvm_event_pool extends uvm_object;
    local uvm_event #(uvm_object) m_pool[string];

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function uvm_event #(uvm_object) get(string name);
      if (!m_pool.exists(name))
        m_pool[name] = new(name);
      return m_pool[name];
    endfunction

    virtual function int num();
      return m_pool.num();
    endfunction

    virtual function void delete(string name);
      if (m_pool.exists(name))
        m_pool.delete(name);
    endfunction

    virtual function int first(ref string name);
      return m_pool.first(name);
    endfunction

    virtual function int next(ref string name);
      return m_pool.next(name);
    endfunction

  endclass

  //=========================================================================
  // Barrier Support
  //=========================================================================
  class uvm_barrier extends uvm_object;
    local int m_threshold;
    local int m_num_waiters;
    local bit m_auto_reset;

    function new(string name = "", int threshold = 0);
      super.new(name);
      m_threshold = threshold;
      m_num_waiters = 0;
      m_auto_reset = 1;
    endfunction

    virtual task wait_for();
      m_num_waiters++;
      if (m_num_waiters >= m_threshold && m_auto_reset)
        m_num_waiters = 0;
    endtask

    virtual function void set_threshold(int threshold);
      m_threshold = threshold;
    endfunction

    virtual function int get_threshold();
      return m_threshold;
    endfunction

    virtual function int get_num_waiters();
      return m_num_waiters;
    endfunction

    virtual function void set_auto_reset(bit value = 1);
      m_auto_reset = value;
    endfunction

    virtual function bit get_auto_reset();
      return m_auto_reset;
    endfunction

    virtual function void reset(bit wakeup = 1);
      m_num_waiters = 0;
    endfunction

    virtual function void cancel();
      m_num_waiters--;
    endfunction

  endclass

  //=========================================================================
  // Pool and Queue Support
  //=========================================================================
  class uvm_pool #(type KEY = int, type T = uvm_void) extends uvm_object;
    local T m_pool[KEY];

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function T get(KEY key);
      if (m_pool.exists(key))
        return m_pool[key];
      return null;
    endfunction

    virtual function void add(KEY key, T item);
      m_pool[key] = item;
    endfunction

    virtual function int num();
      return m_pool.num();
    endfunction

    virtual function void delete(KEY key);
      if (m_pool.exists(key))
        m_pool.delete(key);
    endfunction

    virtual function int exists(KEY key);
      return m_pool.exists(key);
    endfunction

    virtual function int first(ref KEY key);
      return m_pool.first(key);
    endfunction

    virtual function int last(ref KEY key);
      return m_pool.last(key);
    endfunction

    virtual function int next(ref KEY key);
      return m_pool.next(key);
    endfunction

    virtual function int prev(ref KEY key);
      return m_pool.prev(key);
    endfunction

  endclass

  class uvm_queue #(type T = uvm_object) extends uvm_object;
    local T m_queue[$];

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function T get(int index);
      if (index >= 0 && index < m_queue.size())
        return m_queue[index];
      return null;
    endfunction

    virtual function void insert(int index, T item);
      if (index >= m_queue.size())
        m_queue.push_back(item);
      else
        m_queue.insert(index, item);
    endfunction

    virtual function void push_back(T item);
      m_queue.push_back(item);
    endfunction

    virtual function void push_front(T item);
      m_queue.push_front(item);
    endfunction

    virtual function T pop_back();
      if (m_queue.size() > 0)
        return m_queue.pop_back();
      return null;
    endfunction

    virtual function T pop_front();
      if (m_queue.size() > 0)
        return m_queue.pop_front();
      return null;
    endfunction

    virtual function int size();
      return m_queue.size();
    endfunction

    virtual function void delete(int index = -1);
      if (index < 0)
        m_queue.delete();
      else if (index < m_queue.size())
        m_queue.delete(index);
    endfunction

  endclass

  //=========================================================================
  // Heartbeat
  //=========================================================================
  class uvm_heartbeat extends uvm_object;
    function new(string name = "", uvm_component cntxt = null, uvm_objection objection = null);
      super.new(name);
    endfunction

    virtual function void set_mode(uvm_heartbeat_modes mode = UVM_ANY_ACTIVE);
    endfunction

    virtual function void set_heartbeat(uvm_event#(uvm_object) e, ref uvm_component comps[$]);
    endfunction

    virtual function void add(uvm_component comp);
    endfunction

    virtual function void remove(uvm_component comp);
    endfunction

    virtual function void start(uvm_event#(uvm_object) e = null);
    endfunction

    virtual function void stop();
    endfunction

  endclass

  //=========================================================================
  // uvm_cmdline_processor - Command line argument processor
  //=========================================================================
  class uvm_cmdline_processor extends uvm_report_object;
    local static uvm_cmdline_processor m_inst;
    protected string m_argv[$];
    protected string m_plus_argv[$];

    function new(string name = "uvm_cmdline_proc");
      super.new(name);
    endfunction

    static function uvm_cmdline_processor get_inst();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

    virtual function string get_arg(int index);
      if (index >= 0 && index < m_argv.size())
        return m_argv[index];
      return "";
    endfunction

    virtual function int get_arg_count();
      return m_argv.size();
    endfunction

    virtual function void get_args(ref string argv[$]);
      argv = m_argv;
    endfunction

    virtual function void get_plusargs(ref string plusargs[$]);
      plusargs = m_plus_argv;
    endfunction

    virtual function int get_arg_matches(string match, ref string args[$]);
      args.delete();
      foreach (m_argv[i])
        if (m_argv[i].substr(0, match.len()-1) == match)
          args.push_back(m_argv[i]);
      return args.size();
    endfunction

    virtual function int get_arg_value(string match, ref string value);
      foreach (m_argv[i]) begin
        if (m_argv[i].substr(0, match.len()-1) == match) begin
          value = m_argv[i].substr(match.len(), m_argv[i].len()-1);
          return 1;
        end
      end
      return 0;
    endfunction

    virtual function int get_arg_values(string match, ref string values[$]);
      values.delete();
      foreach (m_argv[i]) begin
        if (m_argv[i].substr(0, match.len()-1) == match)
          values.push_back(m_argv[i].substr(match.len(), m_argv[i].len()-1));
      end
      return values.size();
    endfunction

  endclass

  //=========================================================================
  // uvm_report_message - Encapsulates report message data
  //=========================================================================
  // The uvm_report_message class holds all the information about a single
  // report message including severity, id, message text, verbosity, etc.
  //=========================================================================
  class uvm_report_message extends uvm_object;
    protected uvm_severity m_severity;
    protected string m_id;
    protected string m_message;
    protected int m_verbosity;
    protected string m_filename;
    protected int m_line;
    protected string m_context_name;
    protected uvm_action m_action;
    protected UVM_FILE m_file;
    protected uvm_report_object m_report_object;
    protected uvm_report_handler m_report_handler;
    protected uvm_report_server m_report_server;

    function new(string name = "uvm_report_message");
      super.new(name);
      m_severity = UVM_INFO;
      m_verbosity = UVM_MEDIUM;
      m_action = UVM_NO_ACTION;
      m_file = 0;
      m_line = 0;
    endfunction

    // Static factory method
    static function uvm_report_message new_report_message(string name = "uvm_report_message");
      uvm_report_message msg = new(name);
      return msg;
    endfunction

    // Severity
    virtual function uvm_severity get_severity();
      return m_severity;
    endfunction

    virtual function void set_severity(uvm_severity severity);
      m_severity = severity;
    endfunction

    // ID
    virtual function string get_id();
      return m_id;
    endfunction

    virtual function void set_id(string id);
      m_id = id;
    endfunction

    // Message
    virtual function string get_message();
      return m_message;
    endfunction

    virtual function void set_message(string message);
      m_message = message;
    endfunction

    // Verbosity
    virtual function int get_verbosity();
      return m_verbosity;
    endfunction

    virtual function void set_verbosity(int verbosity);
      m_verbosity = verbosity;
    endfunction

    // Filename
    virtual function string get_filename();
      return m_filename;
    endfunction

    virtual function void set_filename(string filename);
      m_filename = filename;
    endfunction

    // Line number
    virtual function int get_line();
      return m_line;
    endfunction

    virtual function void set_line(int line);
      m_line = line;
    endfunction

    // Context name
    virtual function string get_context();
      return m_context_name;
    endfunction

    virtual function void set_context(string context_name);
      m_context_name = context_name;
    endfunction

    // Action
    virtual function uvm_action get_action();
      return m_action;
    endfunction

    virtual function void set_action(uvm_action action);
      m_action = action;
    endfunction

    // File handle
    virtual function UVM_FILE get_file();
      return m_file;
    endfunction

    virtual function void set_file(UVM_FILE file);
      m_file = file;
    endfunction

    // Report object
    virtual function uvm_report_object get_report_object();
      return m_report_object;
    endfunction

    virtual function void set_report_object(uvm_report_object obj);
      m_report_object = obj;
    endfunction

    // Report handler
    virtual function uvm_report_handler get_report_handler();
      return m_report_handler;
    endfunction

    virtual function void set_report_handler(uvm_report_handler handler);
      m_report_handler = handler;
    endfunction

    // Report server
    virtual function uvm_report_server get_report_server();
      return m_report_server;
    endfunction

    virtual function void set_report_server(uvm_report_server server);
      m_report_server = server;
    endfunction

  endclass

  //=========================================================================
  // uvm_report_handler - Report message configuration handler
  //=========================================================================
  // The uvm_report_handler class manages report actions, verbosity levels,
  // and file handles for reporting. Each uvm_report_object has a handler.
  //=========================================================================
  class uvm_report_handler extends uvm_object;
    protected int m_max_verbosity_level = UVM_MEDIUM;
    protected uvm_action severity_actions[uvm_severity];
    protected uvm_action id_actions[string];
    protected int id_verbosities[string];
    protected UVM_FILE severity_file_handles[uvm_severity];
    protected UVM_FILE id_file_handles[string];
    protected UVM_FILE default_file_handle = 0;

    function new(string name = "uvm_report_handler");
      super.new(name);
      initialize();
    endfunction

    virtual function void initialize();
      // Set default actions for each severity
      severity_actions[UVM_INFO] = UVM_DISPLAY;
      severity_actions[UVM_WARNING] = UVM_DISPLAY;
      severity_actions[UVM_ERROR] = UVM_DISPLAY | UVM_COUNT;
      severity_actions[UVM_FATAL] = UVM_DISPLAY | UVM_EXIT;
    endfunction

    // Verbosity level management
    virtual function int get_verbosity_level();
      return m_max_verbosity_level;
    endfunction

    virtual function void set_verbosity_level(int verbosity_level);
      m_max_verbosity_level = verbosity_level;
    endfunction

    // Action management for severities
    virtual function uvm_action get_action(uvm_severity severity, string id);
      if (id_actions.exists(id))
        return id_actions[id];
      if (severity_actions.exists(severity))
        return severity_actions[severity];
      return UVM_NO_ACTION;
    endfunction

    virtual function void set_severity_action(uvm_severity severity, uvm_action action);
      severity_actions[severity] = action;
    endfunction

    virtual function void set_id_action(string id, uvm_action action);
      id_actions[id] = action;
    endfunction

    virtual function void set_severity_id_action(uvm_severity severity, string id, uvm_action action);
      // Simplified: just set the id action
      id_actions[id] = action;
    endfunction

    // Verbosity management for IDs
    virtual function int get_id_verbosity(string id);
      if (id_verbosities.exists(id))
        return id_verbosities[id];
      return m_max_verbosity_level;
    endfunction

    virtual function void set_id_verbosity(string id, int verbosity);
      id_verbosities[id] = verbosity;
    endfunction

    virtual function void set_severity_id_verbosity(uvm_severity severity, string id, int verbosity);
      id_verbosities[id] = verbosity;
    endfunction

    // File handle management
    virtual function UVM_FILE get_file_handle(uvm_severity severity, string id);
      if (id_file_handles.exists(id))
        return id_file_handles[id];
      if (severity_file_handles.exists(severity))
        return severity_file_handles[severity];
      return default_file_handle;
    endfunction

    virtual function void set_severity_file(uvm_severity severity, UVM_FILE file);
      severity_file_handles[severity] = file;
    endfunction

    virtual function void set_id_file(string id, UVM_FILE file);
      id_file_handles[id] = file;
    endfunction

    virtual function void set_severity_id_file(uvm_severity severity, string id, UVM_FILE file);
      id_file_handles[id] = file;
    endfunction

    virtual function void set_default_file(UVM_FILE file);
      default_file_handle = file;
    endfunction

    // Report handler summary
    virtual function void dump_state();
      $display("UVM Report Handler State:");
      $display("  Max verbosity level: %0d", m_max_verbosity_level);
      $display("  Severity actions configured: %0d", severity_actions.num());
      $display("  ID actions configured: %0d", id_actions.num());
      $display("  ID verbosities configured: %0d", id_verbosities.num());
    endfunction

  endclass

  //=========================================================================
  // uvm_report_server - Report server singleton
  //=========================================================================
  class uvm_report_server extends uvm_object;
    local static uvm_report_server m_inst;
    protected int id_count[string];
    protected int severity_count[uvm_severity];
    protected int max_quit_count = 10;
    protected int quit_count = 0;

    function new(string name = "uvm_report_server");
      super.new(name);
    endfunction

    static function uvm_report_server get_server();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

    static function void set_server(uvm_report_server server);
      m_inst = server;
    endfunction

    virtual function int get_severity_count(uvm_severity severity);
      if (severity_count.exists(severity))
        return severity_count[severity];
      return 0;
    endfunction

    virtual function int get_id_count(string id);
      if (id_count.exists(id))
        return id_count[id];
      return 0;
    endfunction

    virtual function void set_severity_count(uvm_severity severity, int count);
      severity_count[severity] = count;
    endfunction

    virtual function void set_id_count(string id, int count);
      id_count[id] = count;
    endfunction

    virtual function void incr_severity_count(uvm_severity severity);
      if (!severity_count.exists(severity))
        severity_count[severity] = 0;
      severity_count[severity]++;
    endfunction

    virtual function void incr_id_count(string id);
      if (!id_count.exists(id))
        id_count[id] = 0;
      id_count[id]++;
    endfunction

    virtual function int get_max_quit_count();
      return max_quit_count;
    endfunction

    virtual function void set_max_quit_count(int count);
      max_quit_count = count;
    endfunction

    virtual function int get_quit_count();
      return quit_count;
    endfunction

    virtual function void incr_quit_count();
      quit_count++;
    endfunction

    virtual function void reset_quit_count();
      quit_count = 0;
    endfunction

    virtual function void reset_severity_counts();
      severity_count.delete();
    endfunction

    // Convenience methods for getting counts by specific severity
    virtual function int get_info_count();
      return get_severity_count(UVM_INFO);
    endfunction

    virtual function int get_warning_count();
      return get_severity_count(UVM_WARNING);
    endfunction

    virtual function int get_error_count();
      return get_severity_count(UVM_ERROR);
    endfunction

    virtual function int get_fatal_count();
      return get_severity_count(UVM_FATAL);
    endfunction

    // Check if quit threshold reached
    virtual function bit is_quit_count_reached();
      return (max_quit_count > 0) && (quit_count >= max_quit_count);
    endfunction

    // Dump the server state for debugging
    virtual function void dump_server_state();
      $display("=== UVM Report Server State ===");
      $display("Max quit count: %0d", max_quit_count);
      $display("Current quit count: %0d", quit_count);
      $display("Severity counts:");
      $display("  UVM_INFO: %0d", get_severity_count(UVM_INFO));
      $display("  UVM_WARNING: %0d", get_severity_count(UVM_WARNING));
      $display("  UVM_ERROR: %0d", get_severity_count(UVM_ERROR));
      $display("  UVM_FATAL: %0d", get_severity_count(UVM_FATAL));
      $display("ID counts tracked: %0d", id_count.num());
      $display("===============================");
    endfunction

    // Summarize report statistics (typically called at end of test)
    virtual function void report_summarize(UVM_FILE file = 0);
      string summary;
      int total_errors;
      total_errors = get_severity_count(UVM_ERROR) + get_severity_count(UVM_FATAL);

      $display("--- UVM Report Summary ---");
      $display("");
      $display("** Report counts by severity");
      $display("UVM_INFO: %0d", get_severity_count(UVM_INFO));
      $display("UVM_WARNING: %0d", get_severity_count(UVM_WARNING));
      $display("UVM_ERROR: %0d", get_severity_count(UVM_ERROR));
      $display("UVM_FATAL: %0d", get_severity_count(UVM_FATAL));
      $display("");
      if (total_errors == 0)
        $display("** Test passed with no errors");
      else
        $display("** Test failed with %0d error(s)", total_errors);
      $display("--------------------------");
    endfunction

    // Compose the message (for custom report formatting)
    virtual function string compose_report_message(
      uvm_report_message report_message,
      string report_object_name = ""
    );
      // Simplified composition - just return the message
      if (report_message != null)
        return report_message.get_message();
      return "";
    endfunction

    // Process the report message through catchers
    virtual function bit process_report_message(uvm_report_message report_message);
      // Default: return 1 to indicate message should be issued
      return 1;
    endfunction

  endclass

  //=========================================================================
  // uvm_report_catcher - Report message catcher for filtering/modifying
  //=========================================================================
  typedef enum {
    THROW,
    CAUGHT
  } uvm_action_type_e;

  class uvm_report_catcher extends uvm_callback;
    local static uvm_report_catcher catchers[$];

    function new(string name = "uvm_report_catcher");
      super.new(name);
    endfunction

    static function void add(uvm_report_catcher catcher);
      catchers.push_back(catcher);
    endfunction

    static function void remove(uvm_report_catcher catcher);
      foreach (catchers[i]) begin
        if (catchers[i] == catcher) begin
          catchers.delete(i);
          return;
        end
      end
    endfunction

    // Override in derived class to implement catching behavior
    virtual function uvm_action_type_e catch_action();
      return THROW;
    endfunction

    // Accessors for current message being caught
    protected uvm_severity m_severity;
    protected string m_id;
    protected string m_message;
    protected int m_verbosity;
    protected string m_filename;
    protected int m_line;

    virtual function uvm_severity get_severity();
      return m_severity;
    endfunction

    virtual function string get_id();
      return m_id;
    endfunction

    virtual function string get_message();
      return m_message;
    endfunction

    virtual function int get_verbosity();
      return m_verbosity;
    endfunction

    virtual function string get_fname();
      return m_filename;
    endfunction

    virtual function int get_line();
      return m_line;
    endfunction

    virtual function void set_severity(uvm_severity severity);
      m_severity = severity;
    endfunction

    virtual function void set_id(string id);
      m_id = id;
    endfunction

    virtual function void set_message(string message);
      m_message = message;
    endfunction

    virtual function void set_verbosity(int verbosity);
      m_verbosity = verbosity;
    endfunction

    // Set the action for the caught message
    virtual function void set_action(uvm_action action);
      // Store action to be applied
    endfunction

    // Get number of registered catchers
    static function int get_catcher_count();
      return catchers.size();
    endfunction

    // Clear all registered catchers
    static function void clear_catchers();
      catchers.delete();
    endfunction

    // Process a message through all registered catchers
    // Returns THROW if message should proceed, CAUGHT if suppressed
    static function uvm_action_type_e process_all_report_catchers(
      uvm_severity severity,
      string id,
      string message,
      int verbosity,
      string filename,
      int line
    );
      uvm_action_type_e result;

      foreach (catchers[i]) begin
        // Set up the catcher with current message info
        catchers[i].m_severity = severity;
        catchers[i].m_id = id;
        catchers[i].m_message = message;
        catchers[i].m_verbosity = verbosity;
        catchers[i].m_filename = filename;
        catchers[i].m_line = line;

        // Call the catcher's catch_action
        result = catchers[i].catch_action();

        // If caught, stop processing and suppress message
        if (result == CAUGHT)
          return CAUGHT;
      end

      // All catchers passed, allow message
      return THROW;
    endfunction

    // Check if message should be issued (called by report methods)
    static function bit do_report(
      uvm_severity severity,
      string id,
      ref string message,
      int verbosity,
      string filename,
      int line
    );
      uvm_action_type_e result;

      // Process through all catchers
      result = process_all_report_catchers(severity, id, message, verbosity, filename, line);

      // Return 1 if message should be issued, 0 if caught
      return (result == THROW);
    endfunction

    // Summarize catcher state (for debugging)
    static function void summarize_catchers();
      $display("UVM Report Catcher Summary:");
      $display("  Registered catchers: %0d", catchers.size());
      foreach (catchers[i]) begin
        $display("    [%0d] %s", i, catchers[i].get_name());
      end
    endfunction

  endclass

  //=========================================================================
  // uvm_default_report_server - Default implementation of report server
  //=========================================================================
  class uvm_default_report_server extends uvm_report_server;
    function new(string name = "uvm_default_report_server");
      super.new(name);
    endfunction
  endclass

  //=========================================================================
  // Global cmdline processor instance
  //=========================================================================
  uvm_cmdline_processor uvm_cmdline_proc = uvm_cmdline_processor::get_inst();

  //=========================================================================
  // uvm_tr_database - Transaction recording database
  //=========================================================================
  class uvm_tr_database extends uvm_object;
    function new(string name = "uvm_tr_database");
      super.new(name);
    endfunction

    virtual function bit open_db();
      return 1;
    endfunction

    virtual function bit close_db();
      return 1;
    endfunction

    virtual function bit is_open();
      return 1;
    endfunction

  endclass

  class uvm_text_tr_database extends uvm_tr_database;
    function new(string name = "uvm_text_tr_database");
      super.new(name);
    endfunction
  endclass

  //=========================================================================
  // uvm_copier - Object copying utility
  //=========================================================================
  class uvm_copier extends uvm_object;
    int unsigned policy = UVM_COPY;

    function new(string name = "copier");
      super.new(name);
    endfunction

    virtual function void copy_object(uvm_object lhs, uvm_object rhs);
      if (lhs != null && rhs != null)
        lhs.copy(rhs);
    endfunction

  endclass

  //=========================================================================
  // uvm_resource_base - Resource base class
  //=========================================================================
  class uvm_resource_base extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction

    virtual function void set_read_only();
    endfunction

    virtual function bit is_read_only();
      return 0;
    endfunction

  endclass

  //=========================================================================
  // uvm_resource_pool - Resource pool
  //=========================================================================
  class uvm_resource_pool extends uvm_object;
    local static uvm_resource_pool m_inst;

    function new(string name = "");
      super.new(name);
    endfunction

    static function uvm_resource_pool get();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

  endclass

  //=========================================================================
  // uvm_visitor - Component visitor pattern
  //=========================================================================
  class uvm_visitor #(type T = uvm_component) extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction

    virtual function void begin_v();
    endfunction

    virtual function void end_v();
    endfunction

    virtual function void visit(T node);
    endfunction

  endclass

  //=========================================================================
  // uvm_coreservice_t - Core service singleton
  //=========================================================================
  virtual class uvm_coreservice_t extends uvm_void;

    pure virtual function uvm_factory get_factory();
    pure virtual function void set_factory(uvm_factory f);
    pure virtual function uvm_report_server get_report_server();
    pure virtual function void set_report_server(uvm_report_server server);
    pure virtual function uvm_tr_database get_default_tr_database();
    pure virtual function void set_default_tr_database(uvm_tr_database db);
    pure virtual function uvm_root get_root();
    pure virtual function void set_component_visitor(uvm_visitor #(uvm_component) v);
    pure virtual function uvm_visitor #(uvm_component) get_component_visitor();

    local static uvm_coreservice_t inst;

    static function uvm_coreservice_t get();
      if (inst == null) begin
        uvm_default_coreservice_t cs = new();
        inst = cs;
      end
      return inst;
    endfunction

    static function void set(uvm_coreservice_t cs);
      inst = cs;
    endfunction

  endclass

  //=========================================================================
  // uvm_default_coreservice_t - Default core service implementation
  //=========================================================================
  class uvm_default_coreservice_t extends uvm_coreservice_t;
    local uvm_factory m_factory;
    local uvm_report_server m_report_server;
    local uvm_tr_database m_tr_database;
    local uvm_root m_root;
    local uvm_visitor #(uvm_component) m_visitor;

    function new();
    endfunction

    virtual function uvm_factory get_factory();
      if (m_factory == null)
        m_factory = uvm_factory::get();
      return m_factory;
    endfunction

    virtual function void set_factory(uvm_factory f);
      m_factory = f;
    endfunction

    virtual function uvm_report_server get_report_server();
      if (m_report_server == null)
        m_report_server = uvm_report_server::get_server();
      return m_report_server;
    endfunction

    virtual function void set_report_server(uvm_report_server server);
      m_report_server = server;
    endfunction

    virtual function uvm_tr_database get_default_tr_database();
      if (m_tr_database == null)
        m_tr_database = new("uvm_text_tr_database");
      return m_tr_database;
    endfunction

    virtual function void set_default_tr_database(uvm_tr_database db);
      m_tr_database = db;
    endfunction

    virtual function uvm_root get_root();
      if (m_root == null)
        m_root = uvm_root::get();
      return m_root;
    endfunction

    virtual function void set_component_visitor(uvm_visitor #(uvm_component) v);
      m_visitor = v;
    endfunction

    virtual function uvm_visitor #(uvm_component) get_component_visitor();
      return m_visitor;
    endfunction

  endclass

  //=========================================================================
  // Additional TLM Ports and Exports commonly used
  //=========================================================================

  // uvm_nonblocking_put_port
  class uvm_nonblocking_put_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function bit try_put(input T t);
      if (m_if != null)
        return m_if.try_put(t);
      return 0;
    endfunction

    virtual function bit can_put();
      if (m_if != null)
        return m_if.can_put();
      return 0;
    endfunction

  endclass

  // uvm_put_port - combined blocking and nonblocking
  class uvm_put_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task put(input T t);
      if (m_if != null)
        m_if.put(t);
    endtask

    virtual function bit try_put(input T t);
      if (m_if != null)
        return m_if.try_put(t);
      return 0;
    endfunction

    virtual function bit can_put();
      if (m_if != null)
        return m_if.can_put();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_get_port
  class uvm_nonblocking_get_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

  endclass

  // uvm_get_port - combined blocking and nonblocking
  class uvm_get_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

  endclass

  // uvm_get_peek_port - combined get and peek port
  class uvm_get_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_blocking_put_export
  class uvm_blocking_put_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task put(input T t);
      if (m_if != null)
        m_if.put(t);
    endtask

  endclass

  // uvm_blocking_get_export
  class uvm_blocking_get_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

  endclass

  // uvm_blocking_peek_export
  class uvm_blocking_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

  endclass

  // uvm_blocking_get_peek_export
  class uvm_blocking_get_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

  endclass

  // uvm_put_export - combined blocking and nonblocking export
  class uvm_put_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task put(input T t);
      if (m_if != null)
        m_if.put(t);
    endtask

    virtual function bit try_put(input T t);
      if (m_if != null)
        return m_if.try_put(t);
      return 0;
    endfunction

    virtual function bit can_put();
      if (m_if != null)
        return m_if.can_put();
      return 0;
    endfunction

  endclass

  // uvm_get_export - combined blocking and nonblocking export
  class uvm_get_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

  endclass

  // uvm_get_peek_export - combined get and peek export
  class uvm_get_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual task get(output T t);
      if (m_if != null)
        m_if.get(t);
    endtask

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  //=========================================================================
  // TLM Implementation (Imp) Classes - Added for AVIP compatibility
  //=========================================================================

  // uvm_blocking_put_imp
  class uvm_blocking_put_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
      // Note: We don't set m_if here because uvm_blocking_put_imp doesn't
      // extend uvm_tlm_if_base. Instead, the put/get/peek methods are
      // implemented directly and port connections should use the provided_by
      // mechanism or direct method calls.
    endfunction

    virtual function string get_type_name();
      return "uvm_blocking_put_imp";
    endfunction

    virtual task put(input T t);
      m_imp.put(t);
    endtask

  endclass

  // uvm_nonblocking_put_imp
  class uvm_nonblocking_put_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_put_imp";
    endfunction

    virtual function bit try_put(input T t);
      return m_imp.try_put(t);
    endfunction

    virtual function bit can_put();
      return m_imp.can_put();
    endfunction

  endclass

  // uvm_blocking_get_imp
  class uvm_blocking_get_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_blocking_get_imp";
    endfunction

    virtual task get(output T t);
      m_imp.get(t);
    endtask

  endclass

  // uvm_nonblocking_get_imp
  class uvm_nonblocking_get_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_get_imp";
    endfunction

    virtual function bit try_get(output T t);
      return m_imp.try_get(t);
    endfunction

    virtual function bit can_get();
      return m_imp.can_get();
    endfunction

  endclass

  // uvm_blocking_peek_imp
  class uvm_blocking_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_blocking_peek_imp";
    endfunction

    virtual task peek(output T t);
      m_imp.peek(t);
    endtask

  endclass

  // uvm_nonblocking_peek_imp
  class uvm_nonblocking_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_peek_imp";
    endfunction

    virtual function bit try_peek(output T t);
      return m_imp.try_peek(t);
    endfunction

    virtual function bit can_peek();
      return m_imp.can_peek();
    endfunction

  endclass

  // uvm_peek_imp - combined blocking and nonblocking peek
  class uvm_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_peek_imp";
    endfunction

    virtual task peek(output T t);
      m_imp.peek(t);
    endtask

    virtual function bit try_peek(output T t);
      return m_imp.try_peek(t);
    endfunction

    virtual function bit can_peek();
      return m_imp.can_peek();
    endfunction

  endclass

  // uvm_blocking_get_peek_imp
  class uvm_blocking_get_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_blocking_get_peek_imp";
    endfunction

    virtual task get(output T t);
      m_imp.get(t);
    endtask

    virtual task peek(output T t);
      m_imp.peek(t);
    endtask

  endclass

  // uvm_nonblocking_get_peek_imp
  class uvm_nonblocking_get_peek_imp #(type T = int, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));
    local IMP m_imp;

    function new(string name, IMP imp);
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1);
      m_imp = imp;
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_get_peek_imp";
    endfunction

    virtual function bit try_get(output T t);
      return m_imp.try_get(t);
    endfunction

    virtual function bit can_get();
      return m_imp.can_get();
    endfunction

    virtual function bit try_peek(output T t);
      return m_imp.try_peek(t);
    endfunction

    virtual function bit can_peek();
      return m_imp.can_peek();
    endfunction

  endclass

  //=========================================================================
  // Additional Nonblocking Ports and Exports for AVIP compatibility
  //=========================================================================

  // uvm_nonblocking_peek_port
  class uvm_nonblocking_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_peek_port";
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_peek_port - combined blocking and nonblocking
  class uvm_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_peek_port";
    endfunction

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_get_peek_port
  class uvm_nonblocking_get_peek_port #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_PORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_get_peek_port";
    endfunction

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_put_export
  class uvm_nonblocking_put_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_put_export";
    endfunction

    virtual function bit try_put(input T t);
      if (m_if != null)
        return m_if.try_put(t);
      return 0;
    endfunction

    virtual function bit can_put();
      if (m_if != null)
        return m_if.can_put();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_get_export
  class uvm_nonblocking_get_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_get_export";
    endfunction

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_peek_export
  class uvm_nonblocking_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_peek_export";
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_peek_export - combined blocking and nonblocking
  class uvm_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_peek_export";
    endfunction

    virtual task peek(output T t);
      if (m_if != null)
        m_if.peek(t);
    endtask

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  // uvm_nonblocking_get_peek_export
  class uvm_nonblocking_get_peek_export #(type T = int)
    extends uvm_port_base #(uvm_tlm_if_base #(T, T));

    function new(string name, uvm_component parent, int min_size = 0, int max_size = 1);
      super.new(name, parent, UVM_EXPORT, min_size, max_size);
    endfunction

    virtual function string get_type_name();
      return "uvm_nonblocking_get_peek_export";
    endfunction

    virtual function bit try_get(output T t);
      if (m_if != null)
        return m_if.try_get(t);
      return 0;
    endfunction

    virtual function bit can_get();
      if (m_if != null)
        return m_if.can_get();
      return 0;
    endfunction

    virtual function bit try_peek(output T t);
      if (m_if != null)
        return m_if.try_peek(t);
      return 0;
    endfunction

    virtual function bit can_peek();
      if (m_if != null)
        return m_if.can_peek();
      return 0;
    endfunction

  endclass

  //=========================================================================
  // uvm_mem - Memory stub
  //=========================================================================
  class uvm_mem extends uvm_object;
    function new(string name = "", longint unsigned size = 0, int unsigned n_bits = 0,
                 string access = "RW", int has_coverage = 0);
      super.new(name);
    endfunction

    virtual task write(output uvm_status_e status, input uvm_reg_addr_t offset,
                       input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
    endtask

    virtual task read(output uvm_status_e status, input uvm_reg_addr_t offset,
                      output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
      value = 0;
    endtask

    virtual function longint unsigned get_size();
      return 0;
    endfunction

    virtual function int unsigned get_n_bits();
      return 0;
    endfunction

    virtual function string get_access(uvm_reg_map map = null);
      return "RW";
    endfunction

  endclass

  //=========================================================================
  // uvm_coverage - Abstract base class for functional coverage
  //=========================================================================
  // This class provides the foundation for functional coverage collection
  // in UVM. It defines the interface for coverage sampling and querying.
  // IEEE 1800.2-2020 Section 19.1
  virtual class uvm_coverage extends uvm_object;
    // Coverage model type for this object
    protected uvm_reg_cvr_t m_coverage_model;
    // Flag to enable/disable coverage
    protected bit m_coverage_enabled;
    // Coverage goal (percentage)
    protected real m_coverage_goal;

    function new(string name = "uvm_coverage");
      super.new(name);
      m_coverage_model = UVM_NO_COVERAGE;
      m_coverage_enabled = 1;
      m_coverage_goal = 100.0;
    endfunction

    virtual function string get_type_name();
      return "uvm_coverage";
    endfunction

    // Enable/disable coverage collection
    virtual function void set_coverage_enabled(bit enabled);
      m_coverage_enabled = enabled;
    endfunction

    virtual function bit get_coverage_enabled();
      return m_coverage_enabled;
    endfunction

    // Set/get coverage goal
    virtual function void set_coverage_goal(real goal);
      m_coverage_goal = goal;
    endfunction

    virtual function real get_coverage_goal();
      return m_coverage_goal;
    endfunction

    // Set/get coverage model
    virtual function uvm_reg_cvr_t set_coverage(uvm_reg_cvr_t is_on);
      uvm_reg_cvr_t prev = m_coverage_model;
      m_coverage_model = is_on;
      return prev;
    endfunction

    virtual function uvm_reg_cvr_t get_coverage(bit is_field = 0);
      return m_coverage_model;
    endfunction

    virtual function bit has_coverage(uvm_reg_cvr_t models);
      return (m_coverage_model & models) != 0;
    endfunction

    // Sample coverage - override in derived classes
    pure virtual function void sample();

    // Get current coverage percentage - override in derived classes
    virtual function real get_coverage_pct();
      return 0.0;
    endfunction

    // Check if coverage goal has been met
    virtual function bit is_coverage_goal_met();
      return get_coverage_pct() >= m_coverage_goal;
    endfunction

  endclass

  //=========================================================================
  // uvm_mem_region - Memory region descriptor for MAM
  //=========================================================================
  // Represents a contiguous memory region managed by the Memory Allocation
  // Manager. Used for dynamic memory allocation in verification.
  class uvm_mem_region extends uvm_object;
    protected longint unsigned m_start_offset;
    protected longint unsigned m_end_offset;
    protected longint unsigned m_len;
    protected int unsigned m_n_bytes;
    protected uvm_mem m_parent;
    protected string m_fname;
    protected int m_lineno;
    protected bit m_is_allocated;

    function new(string name = "uvm_mem_region");
      super.new(name);
      m_start_offset = 0;
      m_end_offset = 0;
      m_len = 0;
      m_n_bytes = 0;
      m_parent = null;
      m_is_allocated = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_mem_region";
    endfunction

    // Initialize the region with bounds
    virtual function void configure(longint unsigned start_offset,
                                    longint unsigned end_offset,
                                    int unsigned n_bytes,
                                    uvm_mem parent);
      m_start_offset = start_offset;
      m_end_offset = end_offset;
      m_len = end_offset - start_offset + 1;
      m_n_bytes = n_bytes;
      m_parent = parent;
      m_is_allocated = 1;
    endfunction

    // Get the start offset
    virtual function longint unsigned get_start_offset();
      return m_start_offset;
    endfunction

    // Get the end offset
    virtual function longint unsigned get_end_offset();
      return m_end_offset;
    endfunction

    // Get the length (number of addresses)
    virtual function longint unsigned get_len();
      return m_len;
    endfunction

    // Get the number of bytes per address
    virtual function int unsigned get_n_bytes();
      return m_n_bytes;
    endfunction

    // Get the parent memory
    virtual function uvm_mem get_memory();
      return m_parent;
    endfunction

    // Check if region is allocated
    virtual function bit is_allocated();
      return m_is_allocated;
    endfunction

    // Release this region
    virtual function void release_region();
      m_is_allocated = 0;
    endfunction

    // Write to this region (offset relative to region start)
    virtual task write(output uvm_status_e status,
                       input longint unsigned offset,
                       input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      if (m_parent != null && offset < m_len) begin
        m_parent.write(status, m_start_offset + offset, value, path, map,
                       parent, prior, extension, fname, lineno);
      end else begin
        status = UVM_NOT_OK;
      end
    endtask

    // Read from this region (offset relative to region start)
    virtual task read(output uvm_status_e status,
                      input longint unsigned offset,
                      output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      if (m_parent != null && offset < m_len) begin
        m_parent.read(status, m_start_offset + offset, value, path, map,
                      parent, prior, extension, fname, lineno);
      end else begin
        status = UVM_NOT_OK;
        value = 0;
      end
    endtask

    virtual function string convert2string();
      return $sformatf("Region[%0h:%0h] len=%0d bytes=%0d allocated=%0b",
                       m_start_offset, m_end_offset, m_len, m_n_bytes, m_is_allocated);
    endfunction

  endclass

  //=========================================================================
  // uvm_mem_mam_policy - Memory allocation policy
  //=========================================================================
  typedef enum {
    UVM_MEM_MAM_GREEDY,   // First fit allocation
    UVM_MEM_MAM_BEST_FIT, // Best fit allocation
    UVM_MEM_MAM_RANDOM    // Random location allocation
  } uvm_mem_mam_policy_e;

  //=========================================================================
  // uvm_mem_mam_cfg - Memory Allocation Manager configuration
  //=========================================================================
  class uvm_mem_mam_cfg extends uvm_object;
    longint unsigned start_offset;
    longint unsigned end_offset;
    int unsigned n_bytes;
    uvm_mem_mam_policy_e mode;
    bit locality;

    function new(string name = "uvm_mem_mam_cfg");
      super.new(name);
      start_offset = 0;
      end_offset = 0;
      n_bytes = 1;
      mode = UVM_MEM_MAM_GREEDY;
      locality = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_mem_mam_cfg";
    endfunction

  endclass

  //=========================================================================
  // uvm_mem_mam - Memory Allocation Manager
  //=========================================================================
  // Provides dynamic memory allocation for verification environments.
  // Tracks allocated and free regions in a memory model.
  // IEEE 1800.2-2020 compliant stub implementation.
  class uvm_mem_mam extends uvm_object;
    protected uvm_mem m_memory;
    protected uvm_mem_mam_cfg m_cfg;
    protected uvm_mem_region m_allocated_regions[$];
    protected longint unsigned m_total_size;
    protected longint unsigned m_allocated_size;

    function new(string name = "", uvm_mem_mam_cfg cfg = null, uvm_mem mem = null);
      super.new(name);
      m_memory = mem;
      m_cfg = cfg;
      m_total_size = 0;
      m_allocated_size = 0;
      if (cfg != null) begin
        m_total_size = cfg.end_offset - cfg.start_offset + 1;
      end
    endfunction

    virtual function string get_type_name();
      return "uvm_mem_mam";
    endfunction

    // Reconfigure the MAM
    virtual function void reconfigure(uvm_mem_mam_cfg cfg = null);
      if (cfg != null) begin
        m_cfg = cfg;
        m_total_size = cfg.end_offset - cfg.start_offset + 1;
      end
    endfunction

    // Reserve a memory region by address
    virtual function uvm_mem_region reserve_region(longint unsigned start_offset,
                                                    int unsigned n_bytes,
                                                    string fname = "",
                                                    int lineno = 0);
      uvm_mem_region region;
      longint unsigned end_offset;

      // Calculate end offset
      end_offset = start_offset + n_bytes - 1;

      // Check bounds
      if (m_cfg != null) begin
        if (start_offset < m_cfg.start_offset || end_offset > m_cfg.end_offset) begin
          return null;
        end
      end

      // Check for overlap with existing regions
      foreach (m_allocated_regions[i]) begin
        if (m_allocated_regions[i].is_allocated()) begin
          longint unsigned r_start = m_allocated_regions[i].get_start_offset();
          longint unsigned r_end = m_allocated_regions[i].get_end_offset();
          if (!(end_offset < r_start || start_offset > r_end)) begin
            // Overlap detected
            return null;
          end
        end
      end

      // Create and configure the region
      region = new($sformatf("region_%0h", start_offset));
      region.configure(start_offset, end_offset, n_bytes, m_memory);
      m_allocated_regions.push_back(region);
      m_allocated_size += n_bytes;

      return region;
    endfunction

    // Request allocation of a memory region
    virtual function uvm_mem_region request_region(int unsigned n_bytes,
                                                    uvm_mem_mam_policy_e alloc_mode = UVM_MEM_MAM_GREEDY,
                                                    string fname = "",
                                                    int lineno = 0);
      uvm_mem_region region;
      longint unsigned start_offset;
      longint unsigned cfg_start;
      longint unsigned cfg_end;

      // Get configuration bounds
      cfg_start = (m_cfg != null) ? m_cfg.start_offset : 0;
      cfg_end = (m_cfg != null) ? m_cfg.end_offset : 64'hFFFFFFFF;

      // Find a free location based on allocation mode
      case (alloc_mode)
        UVM_MEM_MAM_GREEDY: begin
          // First fit: find first available location
          start_offset = cfg_start;
          while (start_offset + n_bytes - 1 <= cfg_end) begin
            bit overlap = 0;
            foreach (m_allocated_regions[i]) begin
              if (m_allocated_regions[i].is_allocated()) begin
                longint unsigned r_start = m_allocated_regions[i].get_start_offset();
                longint unsigned r_end = m_allocated_regions[i].get_end_offset();
                if (!(start_offset + n_bytes - 1 < r_start || start_offset > r_end)) begin
                  overlap = 1;
                  start_offset = r_end + 1;
                  break;
                end
              end
            end
            if (!overlap) break;
          end
          if (start_offset + n_bytes - 1 > cfg_end) begin
            return null;
          end
        end

        UVM_MEM_MAM_BEST_FIT: begin
          // Best fit: find smallest gap that fits
          longint unsigned best_start = 0;
          longint unsigned best_gap = 64'hFFFFFFFFFFFFFFFF;
          bit found = 0;

          // This is a simplified implementation
          start_offset = cfg_start;
          while (start_offset + n_bytes - 1 <= cfg_end) begin
            bit overlap = 0;
            longint unsigned gap_end = cfg_end;

            foreach (m_allocated_regions[i]) begin
              if (m_allocated_regions[i].is_allocated()) begin
                longint unsigned r_start = m_allocated_regions[i].get_start_offset();
                longint unsigned r_end = m_allocated_regions[i].get_end_offset();
                if (r_start > start_offset && r_start < gap_end) begin
                  gap_end = r_start - 1;
                end
                if (!(start_offset + n_bytes - 1 < r_start || start_offset > r_end)) begin
                  overlap = 1;
                  start_offset = r_end + 1;
                  break;
                end
              end
            end

            if (!overlap) begin
              longint unsigned gap_size = gap_end - start_offset + 1;
              if (gap_size >= n_bytes && gap_size < best_gap) begin
                best_gap = gap_size;
                best_start = start_offset;
                found = 1;
              end
              // Move to next gap
              start_offset = gap_end + 1;
            end
          end

          if (!found) return null;
          start_offset = best_start;
        end

        UVM_MEM_MAM_RANDOM: begin
          // Random: try random locations
          int max_tries = 100;
          bit found = 0;
          bit overlap;
          int try_count;

          for (try_count = 0; try_count < max_tries; try_count++) begin
            start_offset = cfg_start + ($urandom() % (cfg_end - cfg_start - n_bytes + 2));
            overlap = 0;
            foreach (m_allocated_regions[i]) begin
              if (m_allocated_regions[i].is_allocated()) begin
                longint unsigned r_start = m_allocated_regions[i].get_start_offset();
                longint unsigned r_end = m_allocated_regions[i].get_end_offset();
                if (!(start_offset + n_bytes - 1 < r_start || start_offset > r_end)) begin
                  overlap = 1;
                  break;
                end
              end
            end
            if (!overlap) begin
              found = 1;
              break;
            end
          end

          if (!found) begin
            // Fall back to greedy
            return request_region(n_bytes, UVM_MEM_MAM_GREEDY, fname, lineno);
          end
        end
      endcase

      return reserve_region(start_offset, n_bytes, fname, lineno);
    endfunction

    // Release a memory region
    virtual function void release_region(uvm_mem_region region);
      if (region != null) begin
        m_allocated_size -= region.get_len();
        region.release_region();
      end
    endfunction

    // Release all allocated regions
    virtual function void release_all_regions();
      foreach (m_allocated_regions[i]) begin
        if (m_allocated_regions[i].is_allocated()) begin
          m_allocated_regions[i].release_region();
        end
      end
      m_allocated_size = 0;
    endfunction

    // Get the managed memory
    virtual function uvm_mem get_memory();
      return m_memory;
    endfunction

    // Get list of allocated regions
    virtual function void get_allocated_regions(ref uvm_mem_region regions[$]);
      regions.delete();
      foreach (m_allocated_regions[i]) begin
        if (m_allocated_regions[i].is_allocated()) begin
          regions.push_back(m_allocated_regions[i]);
        end
      end
    endfunction

    // Get usage statistics
    virtual function longint unsigned get_total_size();
      return m_total_size;
    endfunction

    virtual function longint unsigned get_allocated_size();
      return m_allocated_size;
    endfunction

    virtual function longint unsigned get_available_size();
      return m_total_size - m_allocated_size;
    endfunction

    virtual function string convert2string();
      return $sformatf("MAM: total=%0d allocated=%0d available=%0d regions=%0d",
                       m_total_size, m_allocated_size, get_available_size(),
                       m_allocated_regions.size());
    endfunction

  endclass

  //=========================================================================
  // uvm_coverage_db - Coverage database for collecting coverage info
  //=========================================================================
  // Global coverage database that tracks all coverage objects and provides
  // unified access to coverage data. Singleton pattern.
  class uvm_coverage_db;
    static local uvm_coverage_db m_inst;
    protected uvm_coverage m_coverage_objects[$];
    protected bit m_enabled;

    local function new();
      m_enabled = 1;
    endfunction

    // Get singleton instance
    static function uvm_coverage_db get();
      if (m_inst == null)
        m_inst = new();
      return m_inst;
    endfunction

    // Register a coverage object
    virtual function void register_coverage(uvm_coverage cov);
      if (cov != null)
        m_coverage_objects.push_back(cov);
    endfunction

    // Enable/disable all coverage
    virtual function void set_enabled(bit enabled);
      m_enabled = enabled;
      foreach (m_coverage_objects[i]) begin
        m_coverage_objects[i].set_coverage_enabled(enabled);
      end
    endfunction

    virtual function bit get_enabled();
      return m_enabled;
    endfunction

    // Sample all registered coverage objects
    virtual function void sample_all();
      if (m_enabled) begin
        foreach (m_coverage_objects[i]) begin
          if (m_coverage_objects[i].get_coverage_enabled())
            m_coverage_objects[i].sample();
        end
      end
    endfunction

    // Get overall coverage percentage
    virtual function real get_coverage_pct();
      real total = 0.0;
      int count = 0;
      foreach (m_coverage_objects[i]) begin
        total += m_coverage_objects[i].get_coverage_pct();
        count++;
      end
      return (count > 0) ? (total / count) : 0.0;
    endfunction

    // Get number of registered coverage objects
    virtual function int get_num_coverage_objects();
      return m_coverage_objects.size();
    endfunction

    // Report coverage summary
    virtual function void report();
      $display("=== UVM Coverage Summary ===");
      $display("Total coverage objects: %0d", m_coverage_objects.size());
      $display("Overall coverage: %.2f%%", get_coverage_pct());
      foreach (m_coverage_objects[i]) begin
        $display("  %s: %.2f%%",
                 m_coverage_objects[i].get_name(),
                 m_coverage_objects[i].get_coverage_pct());
      end
    endfunction

  endclass

  //=========================================================================
  // uvm_pair - Pair class for comparator match/mismatch reporting
  //=========================================================================
  class uvm_pair #(type T = uvm_object) extends uvm_object;
    T first;
    T second;

    function new(string name = "uvm_pair");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_pair";
    endfunction

    virtual function void do_copy(uvm_object rhs);
      uvm_pair #(T) rhs_pair;
      super.do_copy(rhs);
      if ($cast(rhs_pair, rhs)) begin
        first = rhs_pair.first;
        second = rhs_pair.second;
      end
    endfunction

    virtual function string convert2string();
      string first_str, second_str;
      if (first != null)
        first_str = first.convert2string();
      else
        first_str = "<null>";
      if (second != null)
        second_str = second.convert2string();
      else
        second_str = "<null>";
      return $sformatf("first: %s, second: %s", first_str, second_str);
    endfunction
  endclass

  //=========================================================================
  // uvm_in_order_comparator - In-order comparator for scoreboard support
  //=========================================================================
  // The in-order comparator compares transactions received on two analysis
  // exports (before_export and after_export). When transactions are received
  // on both ports, they are compared in order. Matches and mismatches are
  // counted and reported.
  //=========================================================================
  class uvm_in_order_comparator #(
    type T = uvm_object,
    type COMP = uvm_comparer,
    type CONVERT = int,
    type PAIR = uvm_pair #(T)
  ) extends uvm_component;

    // Analysis exports for receiving transactions
    uvm_analysis_imp #(T, uvm_in_order_comparator #(T, COMP, CONVERT, PAIR)) before_export;
    uvm_analysis_imp #(T, uvm_in_order_comparator #(T, COMP, CONVERT, PAIR)) after_export;

    // Analysis port for broadcasting match/mismatch pairs
    uvm_analysis_port #(PAIR) pair_ap;

    // Internal FIFOs for buffering transactions
    protected T m_before_fifo[$];
    protected T m_after_fifo[$];

    // Statistics
    protected int unsigned m_matches;
    protected int unsigned m_mismatches;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      before_export = new("before_export", this);
      after_export = new("after_export", this);
      pair_ap = new("pair_ap", this);
      m_matches = 0;
      m_mismatches = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_in_order_comparator";
    endfunction

    // Write function for before_export - buffer transactions and try to compare
    virtual function void write_before(T t);
      m_before_fifo.push_back(t);
      try_compare();
    endfunction

    // Write function for after_export - buffer transactions and try to compare
    virtual function void write_after(T t);
      m_after_fifo.push_back(t);
      try_compare();
    endfunction

    // Default write goes to before_export
    virtual function void write(T t);
      write_before(t);
    endfunction

    // Try to compare if we have items in both FIFOs
    protected virtual function void try_compare();
      T before_item, after_item;
      PAIR pair;
      bit match;

      while (m_before_fifo.size() > 0 && m_after_fifo.size() > 0) begin
        before_item = m_before_fifo.pop_front();
        after_item = m_after_fifo.pop_front();

        // Perform comparison using built-in compare
        match = comp(before_item, after_item);

        // Create pair for reporting
        pair = new("pair");
        pair.first = before_item;
        pair.second = after_item;

        if (match) begin
          m_matches++;
          `uvm_info("CMPMATCH", $sformatf("Match #%0d: %s", m_matches,
            before_item.convert2string()), UVM_MEDIUM)
        end
        else begin
          m_mismatches++;
          `uvm_error("CMPMISMATCH", $sformatf("Mismatch #%0d: before=%s, after=%s",
            m_mismatches, before_item.convert2string(), after_item.convert2string()))
        end

        // Broadcast the pair
        pair_ap.write(pair);
      end
    endfunction

    // Virtual comparison function - can be overridden for custom comparison
    protected virtual function bit comp(T a, T b);
      if (a == null && b == null)
        return 1;
      if (a == null || b == null)
        return 0;
      return a.compare(b);
    endfunction

    // Get match count
    virtual function int unsigned get_matches();
      return m_matches;
    endfunction

    // Get mismatch count
    virtual function int unsigned get_mismatches();
      return m_mismatches;
    endfunction

    // Get total comparison count
    virtual function int unsigned get_total();
      return m_matches + m_mismatches;
    endfunction

    // Flush internal FIFOs
    virtual function void flush();
      m_before_fifo.delete();
      m_after_fifo.delete();
    endfunction

    // Report phase - print statistics
    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("CMPSTATS", $sformatf(
        "Comparator '%s': %0d comparisons, %0d matches, %0d mismatches",
        get_full_name(), get_total(), m_matches, m_mismatches), UVM_LOW)
      if (m_mismatches > 0)
        `uvm_error("CMPERR", $sformatf(
          "Comparator '%s' completed with %0d mismatches", get_full_name(), m_mismatches))
      if (m_before_fifo.size() > 0)
        `uvm_warning("CMPWARN", $sformatf(
          "Comparator '%s' has %0d unmatched 'before' items",
          get_full_name(), m_before_fifo.size()))
      if (m_after_fifo.size() > 0)
        `uvm_warning("CMPWARN", $sformatf(
          "Comparator '%s' has %0d unmatched 'after' items",
          get_full_name(), m_after_fifo.size()))
    endfunction

  endclass

  //=========================================================================
  // uvm_in_order_built_in_comparator - Uses built-in compare() method
  //=========================================================================
  // This is a simplified version of the in-order comparator that uses
  // the built-in compare() method of uvm_object directly.
  //=========================================================================
  class uvm_in_order_built_in_comparator #(type T = uvm_object)
    extends uvm_in_order_comparator #(T);

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "uvm_in_order_built_in_comparator";
    endfunction

    // Use the default compare() method
    protected virtual function bit comp(T a, T b);
      if (a == null && b == null)
        return 1;
      if (a == null || b == null)
        return 0;
      return a.compare(b);
    endfunction

  endclass

  //=========================================================================
  // uvm_algorithmic_comparator - Comparator with transformer function
  //=========================================================================
  // The algorithmic comparator allows a custom transformation to be applied
  // to the 'before' transaction before comparison. This is useful when the
  // expected output differs from the input in a predictable way.
  //
  // Type parameters:
  //   BEFORE - Type of transaction on before_export
  //   AFTER  - Type of transaction on after_export
  //   TRANSFORMER - Class that provides transform() method
  //=========================================================================
  class uvm_algorithmic_comparator #(
    type BEFORE = uvm_object,
    type AFTER = uvm_object,
    type TRANSFORMER = uvm_object
  ) extends uvm_component;

    // Analysis exports for receiving transactions
    uvm_analysis_imp #(BEFORE, uvm_algorithmic_comparator #(BEFORE, AFTER, TRANSFORMER)) before_export;
    uvm_analysis_imp #(AFTER, uvm_algorithmic_comparator #(BEFORE, AFTER, TRANSFORMER)) after_export;

    // Analysis port for broadcasting match/mismatch pairs
    uvm_analysis_port #(uvm_pair #(AFTER)) pair_ap;

    // Internal FIFOs for buffering transactions
    protected BEFORE m_before_fifo[$];
    protected AFTER m_after_fifo[$];

    // Transformer object
    TRANSFORMER m_transformer;

    // Statistics
    protected int unsigned m_matches;
    protected int unsigned m_mismatches;

    function new(string name, uvm_component parent, TRANSFORMER transformer = null);
      super.new(name, parent);
      before_export = new("before_export", this);
      after_export = new("after_export", this);
      pair_ap = new("pair_ap", this);
      m_transformer = transformer;
      m_matches = 0;
      m_mismatches = 0;
    endfunction

    virtual function string get_type_name();
      return "uvm_algorithmic_comparator";
    endfunction

    // Set the transformer object
    virtual function void set_transformer(TRANSFORMER transformer);
      m_transformer = transformer;
    endfunction

    // Get the transformer object
    virtual function TRANSFORMER get_transformer();
      return m_transformer;
    endfunction

    // Write function for before_export - buffer transactions and try to compare
    virtual function void write_before(BEFORE t);
      m_before_fifo.push_back(t);
      try_compare();
    endfunction

    // Write function for after_export - buffer transactions and try to compare
    virtual function void write_after(AFTER t);
      m_after_fifo.push_back(t);
      try_compare();
    endfunction

    // Default write goes to before_export
    virtual function void write(BEFORE t);
      write_before(t);
    endfunction

    // Try to compare if we have items in both FIFOs
    protected virtual function void try_compare();
      BEFORE before_item;
      AFTER after_item, transformed_item;
      uvm_pair #(AFTER) pair;
      bit match;

      while (m_before_fifo.size() > 0 && m_after_fifo.size() > 0) begin
        before_item = m_before_fifo.pop_front();
        after_item = m_after_fifo.pop_front();

        // Transform the before item
        transformed_item = transform(before_item);

        // Perform comparison
        match = comp(transformed_item, after_item);

        // Create pair for reporting
        pair = new("pair");
        pair.first = transformed_item;
        pair.second = after_item;

        if (match) begin
          m_matches++;
          `uvm_info("CMPMATCH", $sformatf("Match #%0d", m_matches), UVM_MEDIUM)
        end
        else begin
          m_mismatches++;
          `uvm_error("CMPMISMATCH", $sformatf("Mismatch #%0d: expected=%s, actual=%s",
            m_mismatches,
            (transformed_item != null) ? transformed_item.convert2string() : "<null>",
            (after_item != null) ? after_item.convert2string() : "<null>"))
        end

        // Broadcast the pair
        pair_ap.write(pair);
      end
    endfunction

    // Transform function - override in derived class or set transformer
    protected virtual function AFTER transform(BEFORE t);
      // Default implementation: try to cast directly
      // In real usage, the transformer's transform() method would be called
      AFTER result;
      if (m_transformer != null) begin
        // Call transformer's transform method if it exists
        // Since we can't call arbitrary methods, users should override this
        `uvm_info("TRANSFORM", "Using default transform (direct cast)", UVM_DEBUG)
      end
      if (!$cast(result, t))
        result = null;
      return result;
    endfunction

    // Comparison function - can be overridden
    protected virtual function bit comp(AFTER a, AFTER b);
      if (a == null && b == null)
        return 1;
      if (a == null || b == null)
        return 0;
      return a.compare(b);
    endfunction

    // Get match count
    virtual function int unsigned get_matches();
      return m_matches;
    endfunction

    // Get mismatch count
    virtual function int unsigned get_mismatches();
      return m_mismatches;
    endfunction

    // Get total comparison count
    virtual function int unsigned get_total();
      return m_matches + m_mismatches;
    endfunction

    // Flush internal FIFOs
    virtual function void flush();
      m_before_fifo.delete();
      m_after_fifo.delete();
    endfunction

    // Report phase - print statistics
    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("CMPSTATS", $sformatf(
        "Algorithmic Comparator '%s': %0d comparisons, %0d matches, %0d mismatches",
        get_full_name(), get_total(), m_matches, m_mismatches), UVM_LOW)
      if (m_mismatches > 0)
        `uvm_error("CMPERR", $sformatf(
          "Algorithmic Comparator '%s' completed with %0d mismatches", get_full_name(), m_mismatches))
      if (m_before_fifo.size() > 0)
        `uvm_warning("CMPWARN", $sformatf(
          "Algorithmic Comparator '%s' has %0d unmatched 'before' items",
          get_full_name(), m_before_fifo.size()))
      if (m_after_fifo.size() > 0)
        `uvm_warning("CMPWARN", $sformatf(
          "Algorithmic Comparator '%s' has %0d unmatched 'after' items",
          get_full_name(), m_after_fifo.size()))
    endfunction

  endclass

  //=========================================================================
  // run_test task (module-level task)
  //=========================================================================
  // Note: run_test function is already defined in uvm_pkg as a function.
  // In real UVM this is a task. We provide a task version for compatibility.
  task automatic uvm_run_test(string test_name = "");
    uvm_coreservice_t cs = uvm_coreservice_t::get();
    uvm_root top = cs.get_root();
    $display("[UVM] Running test: %s", test_name);
  endtask

  //=========================================================================
  // uvm_get_report_object - Global function to get report object
  //=========================================================================
  // Returns the uvm_root singleton, which is a uvm_report_object.
  // This avoids the recursion issue present in the real UVM library where
  // uvm_get_report_object -> uvm_coreservice_t::get -> uvm_init -> ...
  function uvm_report_object uvm_get_report_object();
    return uvm_root::get();
  endfunction

endpackage

`endif // UVM_PKG_SV
