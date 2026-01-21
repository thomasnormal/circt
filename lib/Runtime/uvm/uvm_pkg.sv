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
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`ifndef UVM_PKG_SV
`define UVM_PKG_SV

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
  typedef class uvm_tr_database;
  typedef class uvm_report_server;
  typedef class uvm_resource_base;
  typedef class uvm_resource_pool;
  typedef class uvm_default_coreservice_t;

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

  // Run test function
  function automatic void run_test(string test_name = "");
    $display("[UVM] run_test called with: %s", test_name);
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

    function new(string name = "uvm_phase");
      super.new(name);
      m_phase_name = name;
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

    // Objection methods
    virtual function void raise_objection(uvm_object obj = null, string description = "", int count = 1);
      // Stub implementation
    endfunction

    virtual function void drop_objection(uvm_object obj = null, string description = "", int count = 1);
      // Stub implementation
    endfunction

    virtual function uvm_objection get_objection();
      return null;
    endfunction

    // Phase state
    virtual function bit is_before(uvm_phase phase);
      return 0;
    endfunction

    virtual function bit is_after(uvm_phase phase);
      return 0;
    endfunction

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

  //=========================================================================
  // uvm_objection - Objection mechanism
  //=========================================================================
  class uvm_objection extends uvm_object;
    protected int m_count;

    function new(string name = "uvm_objection");
      super.new(name);
      m_count = 0;
    endfunction

    virtual function void raise_objection(uvm_object obj = null, string description = "", int count = 1);
      m_count += count;
    endfunction

    virtual function void drop_objection(uvm_object obj = null, string description = "", int count = 1);
      m_count -= count;
      if (m_count <= 0) begin
        m_count = 0;
      end
    endfunction

    virtual function void set_drain_time(uvm_object obj = null, time drain);
    endfunction

    virtual function int get_objection_count(uvm_object obj = null);
      return m_count;
    endfunction

    virtual task wait_for(uvm_objection_event objt_event, uvm_object obj = null);
    endtask

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

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function void uvm_report_info(string id, string message,
                                          int verbosity = UVM_MEDIUM,
                                          string filename = "", int line = 0);
      if (verbosity <= m_verbosity)
        $display("[%0t] UVM_INFO %s (%s) %s", $time, get_full_name(), id, message);
    endfunction

    virtual function void uvm_report_warning(string id, string message,
                                             int verbosity = UVM_MEDIUM,
                                             string filename = "", int line = 0);
      $display("[%0t] UVM_WARNING %s (%s) %s", $time, get_full_name(), id, message);
    endfunction

    virtual function void uvm_report_error(string id, string message,
                                           int verbosity = UVM_LOW,
                                           string filename = "", int line = 0);
      $display("[%0t] UVM_ERROR %s (%s) %s", $time, get_full_name(), id, message);
    endfunction

    virtual function void uvm_report_fatal(string id, string message,
                                           int verbosity = UVM_NONE,
                                           string filename = "", int line = 0);
      $display("[%0t] UVM_FATAL %s (%s) %s", $time, get_full_name(), id, message);
    endfunction

    virtual function void set_report_verbosity_level(int verbosity);
      m_verbosity = verbosity;
    endfunction

    virtual function int get_report_verbosity_level();
      return m_verbosity;
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

  endclass

  //=========================================================================
  // uvm_root - Top of component hierarchy
  //=========================================================================
  class uvm_root extends uvm_component;
    local static uvm_root m_inst;
    protected uvm_factory m_factory;

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

    virtual task run_test(string test_name = "");
      $display("[UVM] Running test: %s", test_name);
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

    virtual task wait_for_grant(int item_priority = -1, bit lock_request = 0);
      // Stub - would wait for sequencer grant in real UVM
    endtask

    virtual function void send_request(uvm_sequence_item request, bit rerandomize = 0);
      // Stub
    endfunction

    virtual task wait_for_item_done(int transaction_id = -1);
    endtask

    // Item execution methods
    virtual task start_item(uvm_sequence_item item, int set_priority = -1,
                            uvm_sequencer_base sequencer = null);
      if (sequencer == null)
        sequencer = m_sequencer;
      item.set_item_context(this, sequencer);
      wait_for_grant();
    endtask

    virtual task finish_item(uvm_sequence_item item, int set_priority = -1);
      send_request(item);
      wait_for_item_done();
    endtask

    virtual function int unsigned get_priority();
      return m_priority;
    endfunction

    virtual function void set_priority(int unsigned value);
      m_priority = value;
    endfunction

    virtual function bit is_blocked();
      return 0;
    endfunction

    virtual function bit has_lock();
      return 0;
    endfunction

    virtual task lock(uvm_sequencer_base sequencer = null);
    endtask

    virtual task unlock(uvm_sequencer_base sequencer = null);
    endtask

    virtual task grab(uvm_sequencer_base sequencer = null);
    endtask

    virtual task ungrab(uvm_sequencer_base sequencer = null);
    endtask

    // Response queue
    virtual function void use_response_handler(bit enable);
    endfunction

    virtual function uvm_sequence_item get_base_response(int transaction_id = -1);
      return null;
    endfunction

    virtual function void set_response_queue_error_report_disabled(bit value);
    endfunction

    virtual function bit get_response_queue_error_report_disabled();
      return 0;
    endfunction

    virtual function void set_response_queue_depth(int value);
    endfunction

    virtual function int get_response_queue_depth();
      return -1;
    endfunction

  endclass

  //=========================================================================
  // uvm_sequence - Parameterized sequence class
  //=========================================================================
  class uvm_sequence #(type REQ = uvm_sequence_item, type RSP = REQ) extends uvm_sequence_base;
    REQ req;
    RSP rsp;

    protected RSP response_queue[$];

    function new(string name = "uvm_sequence");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequence";
    endfunction

    // Get response from response queue
    virtual task get_response(output RSP response, input int transaction_id = -1);
      response = null;
    endtask

    virtual function void put_response(RSP response);
    endfunction

  endclass

  //=========================================================================
  // uvm_sequencer_base - Base class for sequencers
  //=========================================================================
  class uvm_sequencer_base extends uvm_component;
    protected int m_arbitration;
    protected uvm_sequence_base m_sequences[$];
    protected int m_lock_arb_size;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      m_arbitration = UVM_SEQ_ARB_FIFO;
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

    virtual function int is_blocked(uvm_sequence_base seq);
      return 0;
    endfunction

    virtual function bit is_locked(uvm_sequence_base seq);
      return 0;
    endfunction

    virtual task wait_for_grant(uvm_sequence_base seq, int item_priority = -1, bit lock = 0);
    endtask

    virtual task wait_for_sequences();
    endtask

    virtual function int user_priority_arbitration(int avail_sequences[$]);
      return 0;
    endfunction

    virtual function void stop_sequences();
    endfunction

    virtual task lock(uvm_sequence_base seq);
    endtask

    virtual task unlock(uvm_sequence_base seq);
    endtask

    virtual task grab(uvm_sequence_base seq);
    endtask

    virtual task ungrab(uvm_sequence_base seq);
    endtask

    virtual function void set_max_zero_time_wait_relevant_count(int value);
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

    function new(string name, uvm_component parent);
      super.new(name, parent);
      seq_item_export = new("seq_item_export", this);
    endfunction

    virtual function string get_type_name();
      return "uvm_sequencer_param_base";
    endfunction

    virtual task get_next_item(output REQ t);
      t = null;
    endtask

    virtual task try_next_item(output REQ t);
      t = null;
    endtask

    virtual function void item_done(RSP rsp = null);
    endfunction

    virtual task put(RSP rsp);
    endtask

    virtual task get(output RSP rsp);
      rsp = null;
    endtask

    virtual function bit has_do_available();
      return 0;
    endfunction

    virtual task peek(output REQ t);
      t = null;
    endtask

    virtual function void send_request(uvm_sequence_base sequence_ptr, uvm_sequence_item t, bit rerandomize = 0);
    endfunction

    virtual task get_base_response(output uvm_sequence_item rsp, input int transaction_id = -1);
      rsp = null;
    endtask

    virtual function void put_response(uvm_sequence_item rsp);
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
  class uvm_seq_item_pull_port #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP));

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
  class uvm_seq_item_pull_imp #(type REQ = uvm_sequence_item, type RSP = REQ, type IMP = uvm_component)
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP));
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
  class uvm_seq_item_pull_export #(type REQ = uvm_sequence_item, type RSP = REQ)
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP));

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
      t = null;
    endtask

    virtual function bit try_get(output T t);
      t = null;
      return 0;
    endfunction

    virtual function bit can_get();
      return !is_empty();
    endfunction

    virtual task peek(output T t);
      t = null;
    endtask

    virtual function bit try_peek(output T t);
      t = null;
      return 0;
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
  // uvm_config_db - Configuration database
  //=========================================================================
  class uvm_config_db #(type T = int);

    static local T m_db[string];

    static function void set(uvm_component cntxt, string inst_name,
                             string field_name, T value);
      string key;
      if (cntxt != null)
        key = {cntxt.get_full_name(), ".", inst_name, ".", field_name};
      else
        key = {inst_name, ".", field_name};
      m_db[key] = value;
    endfunction

    static function bit get(uvm_component cntxt, string inst_name,
                            string field_name, ref T value);
      string key;
      if (cntxt != null)
        key = {cntxt.get_full_name(), ".", inst_name, ".", field_name};
      else
        key = {inst_name, ".", field_name};

      if (m_db.exists(key)) begin
        value = m_db[key];
        return 1;
      end
      // Try without context
      key = {inst_name, ".", field_name};
      if (m_db.exists(key)) begin
        value = m_db[key];
        return 1;
      end
      // Try just field name
      if (m_db.exists(field_name)) begin
        value = m_db[field_name];
        return 1;
      end
      return 0;
    endfunction

    static function bit exists(uvm_component cntxt, string inst_name,
                               string field_name);
      string key;
      if (cntxt != null)
        key = {cntxt.get_full_name(), ".", inst_name, ".", field_name};
      else
        key = {inst_name, ".", field_name};
      return m_db.exists(key);
    endfunction

    static function void wait_modified(uvm_component cntxt, string inst_name, string field_name);
      // Stub - would wait for value to be modified in real UVM
    endfunction

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
      if (m_type_overrides.exists(requested_type_name))
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
      if (m_type_overrides.exists(requested_type_name))
        override_type = m_type_overrides[requested_type_name];

      if (m_type_names.exists(override_type))
        return m_type_names[override_type].create_component(name, parent);
      return null;
    endfunction

    virtual function void set_type_override_by_name(string original_type_name,
                                                    string override_type_name,
                                                    bit replace = 1);
      m_type_overrides[original_type_name] = override_type_name;
    endfunction

    virtual function void set_inst_override_by_name(string original_type_name,
                                                    string override_type_name,
                                                    string full_inst_path);
      m_inst_overrides[{full_inst_path, ":", original_type_name}] = override_type_name;
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
  // RAL (Register Abstraction Layer) Stubs
  //=========================================================================

  // uvm_reg_adapter - Register adapter base
  class uvm_reg_adapter extends uvm_object;
    bit supports_byte_enable = 0;
    bit provides_responses = 0;

    function new(string name = "");
      super.new(name);
    endfunction

    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      return null;
    endfunction

    virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
    endfunction

  endclass

  // uvm_reg - Register base class stub
  class uvm_reg extends uvm_object;
    function new(string name = "", int n_bits = 0, int has_coverage = 0);
      super.new(name);
    endfunction

    virtual task write(output uvm_status_e status, input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
    endtask

    virtual task read(output uvm_status_e status, output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
      value = 0;
    endtask

  endclass

  // uvm_reg_block stub
  class uvm_reg_block extends uvm_object;
    function new(string name = "", int has_coverage = 0);
      super.new(name);
    endfunction

    virtual function void lock_model();
    endfunction

    virtual function void set_hdl_path_root(string path, string kind = "RTL");
    endfunction

  endclass

  // uvm_reg_map stub
  class uvm_reg_map extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction

    virtual function void set_sequencer(uvm_sequencer_base sequencer, uvm_reg_adapter adapter = null);
    endfunction

    virtual function void set_auto_predict(bit on = 1);
    endfunction

  endclass

  //=========================================================================
  // Callback Support
  //=========================================================================
  class uvm_callback extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction
  endclass

  class uvm_callbacks #(type T = uvm_object, type CB = uvm_callback);
    static function void add(T obj, CB cb, uvm_apprepend ordering = UVM_APPEND);
    endfunction

    static function void delete(T obj, CB cb);
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
  // uvm_reg_field - Register field stub
  //=========================================================================
  class uvm_reg_field extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction

    virtual function void configure(uvm_reg parent,
                                    int unsigned size,
                                    int unsigned lsb_pos,
                                    string access,
                                    bit volatile_,
                                    uvm_reg_data_t reset,
                                    bit has_reset,
                                    bit is_rand,
                                    bit individually_accessible);
    endfunction

    virtual task write(output uvm_status_e status, input uvm_reg_data_t value,
                       input uvm_path_e path = UVM_DEFAULT_PATH,
                       input uvm_reg_map map = null,
                       input uvm_sequence_base parent = null,
                       input int prior = -1,
                       input uvm_object extension = null,
                       input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
    endtask

    virtual task read(output uvm_status_e status, output uvm_reg_data_t value,
                      input uvm_path_e path = UVM_DEFAULT_PATH,
                      input uvm_reg_map map = null,
                      input uvm_sequence_base parent = null,
                      input int prior = -1,
                      input uvm_object extension = null,
                      input string fname = "", input int lineno = 0);
      status = UVM_IS_OK;
      value = 0;
    endtask

    virtual function uvm_reg_data_t get();
      return 0;
    endfunction

    virtual function void set(uvm_reg_data_t value);
    endfunction

    virtual function uvm_reg_data_t get_reset(string kind = "HARD");
      return 0;
    endfunction

    virtual function int unsigned get_n_bits();
      return 0;
    endfunction

    virtual function int unsigned get_lsb_pos();
      return 0;
    endfunction

    virtual function string get_access(uvm_reg_map map = null);
      return "RW";
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
  // run_test task (module-level task)
  //=========================================================================
  // Note: run_test function is already defined in uvm_pkg as a function.
  // In real UVM this is a task. We provide a task version for compatibility.
  task automatic uvm_run_test(string test_name = "");
    uvm_coreservice_t cs = uvm_coreservice_t::get();
    uvm_root top = cs.get_root();
    $display("[UVM] Running test: %s", test_name);
  endtask

endpackage

`endif // UVM_PKG_SV
