//===----------------------------------------------------------------------===//
// UVM Macros Stubs for CIRCT/Slang Compilation
//===----------------------------------------------------------------------===//
//
// This file provides minimal UVM macro definitions that allow testbench code
// using UVM macros to compile with circt-verilog. These are stubs - they
// provide the interface but not the full UVM functionality.
//
//===----------------------------------------------------------------------===//

`ifndef UVM_MACROS_SVH
`define UVM_MACROS_SVH

//===----------------------------------------------------------------------===//
// Verbosity Levels
//===----------------------------------------------------------------------===//

typedef enum {
  UVM_NONE   = 0,
  UVM_LOW    = 100,
  UVM_MEDIUM = 200,
  UVM_HIGH   = 300,
  UVM_FULL   = 400,
  UVM_DEBUG  = 500
} uvm_verbosity;

//===----------------------------------------------------------------------===//
// Message Macros
//===----------------------------------------------------------------------===//

// UVM_INFO - Report an informational message
`define uvm_info(ID, MSG, VERBOSITY) \
  begin \
    if (uvm_report_enabled(VERBOSITY, UVM_INFO, ID)) \
      uvm_report_info(ID, MSG, VERBOSITY, `uvm_file, `uvm_line); \
  end

// UVM_WARNING - Report a warning message
`define uvm_warning(ID, MSG) \
  begin \
    if (uvm_report_enabled(UVM_NONE, UVM_WARNING, ID)) \
      uvm_report_warning(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

// UVM_ERROR - Report an error message
`define uvm_error(ID, MSG) \
  begin \
    if (uvm_report_enabled(UVM_NONE, UVM_ERROR, ID)) \
      uvm_report_error(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

// UVM_FATAL - Report a fatal error and exit simulation
`define uvm_fatal(ID, MSG) \
  begin \
    if (uvm_report_enabled(UVM_NONE, UVM_FATAL, ID)) \
      uvm_report_fatal(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

// Simple versions that directly print (for stub compatibility)
`define uvm_info_simple(ID, MSG, VERBOSITY) \
  $display("[UVM_INFO] %s: %s", ID, MSG)

`define uvm_warning_simple(ID, MSG) \
  $display("[UVM_WARNING] %s: %s", ID, MSG)

`define uvm_error_simple(ID, MSG) \
  $display("[UVM_ERROR] %s: %s", ID, MSG)

`define uvm_fatal_simple(ID, MSG) \
  begin \
    $display("[UVM_FATAL] %s: %s", ID, MSG); \
    $finish; \
  end

// File and line macros
`define uvm_file `__FILE__
`define uvm_line `__LINE__

//===----------------------------------------------------------------------===//
// Factory Registration Macros
//===----------------------------------------------------------------------===//

// UVM_COMPONENT_UTILS - Register a component with the factory
`define uvm_component_utils(T) \
  typedef uvm_component_registry #(T, `"T`") type_id; \
  static function type_id get_type(); \
    return type_id::get(); \
  endfunction \
  virtual function string get_type_name(); \
    return `"T`"; \
  endfunction

// UVM_COMPONENT_UTILS_BEGIN/END - For field automation
`define uvm_component_utils_begin(T) \
  `uvm_component_utils(T)

`define uvm_component_utils_end

// UVM_OBJECT_UTILS - Register an object with the factory
`define uvm_object_utils(T) \
  typedef uvm_object_registry #(T, `"T`") type_id; \
  static function type_id get_type(); \
    return type_id::get(); \
  endfunction \
  virtual function string get_type_name(); \
    return `"T`"; \
  endfunction \
  virtual function uvm_object create(string name = ""); \
    T tmp = new(name); \
    return tmp; \
  endfunction

// UVM_OBJECT_UTILS_BEGIN/END - For field automation
`define uvm_object_utils_begin(T) \
  `uvm_object_utils(T)

`define uvm_object_utils_end

// Parameterized versions
`define uvm_component_param_utils(T) \
  `uvm_component_utils(T)

`define uvm_object_param_utils(T) \
  `uvm_object_utils(T)

//===----------------------------------------------------------------------===//
// Field Automation Macros (stubs - no actual automation)
//===----------------------------------------------------------------------===//

`define uvm_field_int(ARG, FLAG)
`define uvm_field_object(ARG, FLAG)
`define uvm_field_string(ARG, FLAG)
`define uvm_field_enum(T, ARG, FLAG)
`define uvm_field_array_int(ARG, FLAG)
`define uvm_field_array_object(ARG, FLAG)
`define uvm_field_array_string(ARG, FLAG)
`define uvm_field_queue_int(ARG, FLAG)
`define uvm_field_queue_object(ARG, FLAG)
`define uvm_field_queue_string(ARG, FLAG)
`define uvm_field_aa_int_string(ARG, FLAG)
`define uvm_field_aa_object_string(ARG, FLAG)
`define uvm_field_aa_string_string(ARG, FLAG)
`define uvm_field_aa_int_int(ARG, FLAG)
`define uvm_field_sarray_int(ARG, FLAG)
`define uvm_field_sarray_object(ARG, FLAG)
`define uvm_field_real(ARG, FLAG)
`define uvm_field_event(ARG, FLAG)

// Field flags
`define UVM_ALL_ON       'hFFFF
`define UVM_DEFAULT      'h0001
`define UVM_NOCOMPARE    'h0002
`define UVM_NOPRINT      'h0004
`define UVM_NOPACK       'h0008
`define UVM_NOCOPY       'h0010
`define UVM_READONLY     'h0020
`define UVM_REFERENCE    'h0040

//===----------------------------------------------------------------------===//
// Sequence Macros
//===----------------------------------------------------------------------===//

`define uvm_do(SEQ_OR_ITEM) \
  begin \
    uvm_sequence_base __seq; \
    `uvm_create(SEQ_OR_ITEM) \
    `uvm_send(SEQ_OR_ITEM) \
  end

`define uvm_do_with(SEQ_OR_ITEM, CONSTRAINTS) \
  begin \
    `uvm_create(SEQ_OR_ITEM) \
    if (!SEQ_OR_ITEM.randomize() with CONSTRAINTS) \
      `uvm_warning("RNDFLD", "Randomization failed") \
    `uvm_send(SEQ_OR_ITEM) \
  end

`define uvm_do_on(SEQ_OR_ITEM, SEQR) \
  begin \
    `uvm_create_on(SEQ_OR_ITEM, SEQR) \
    `uvm_send(SEQ_OR_ITEM) \
  end

`define uvm_do_on_with(SEQ_OR_ITEM, SEQR, CONSTRAINTS) \
  begin \
    `uvm_create_on(SEQ_OR_ITEM, SEQR) \
    if (!SEQ_OR_ITEM.randomize() with CONSTRAINTS) \
      `uvm_warning("RNDFLD", "Randomization failed") \
    `uvm_send(SEQ_OR_ITEM) \
  end

`define uvm_do_on_pri(SEQ_OR_ITEM, SEQR, PRI) \
  `uvm_do_on(SEQ_OR_ITEM, SEQR)

`define uvm_do_on_pri_with(SEQ_OR_ITEM, SEQR, PRI, CONSTRAINTS) \
  `uvm_do_on_with(SEQ_OR_ITEM, SEQR, CONSTRAINTS)

`define uvm_create(SEQ_OR_ITEM) \
  SEQ_OR_ITEM = new();

`define uvm_create_on(SEQ_OR_ITEM, SEQR) \
  SEQ_OR_ITEM = new();

`define uvm_send(SEQ_OR_ITEM) \
  begin \
    start_item(SEQ_OR_ITEM); \
    finish_item(SEQ_OR_ITEM); \
  end

`define uvm_send_pri(SEQ_OR_ITEM, PRI) \
  `uvm_send(SEQ_OR_ITEM)

`define uvm_rand_send(SEQ_OR_ITEM) \
  begin \
    start_item(SEQ_OR_ITEM); \
    void'(SEQ_OR_ITEM.randomize()); \
    finish_item(SEQ_OR_ITEM); \
  end

`define uvm_rand_send_with(SEQ_OR_ITEM, CONSTRAINTS) \
  begin \
    start_item(SEQ_OR_ITEM); \
    void'(SEQ_OR_ITEM.randomize() with CONSTRAINTS); \
    finish_item(SEQ_OR_ITEM); \
  end

//===----------------------------------------------------------------------===//
// Object Creation Macros
//===----------------------------------------------------------------------===//

`define uvm_object_create(T) \
  T::type_id::create

`define uvm_component_create(T) \
  T::type_id::create

//===----------------------------------------------------------------------===//
// Declare Macros (for sequences)
//===----------------------------------------------------------------------===//

`define uvm_declare_p_sequencer(SEQUENCER) \
  SEQUENCER p_sequencer; \
  virtual function void m_set_p_sequencer(); \
    super.m_set_p_sequencer(); \
    if (!$cast(p_sequencer, m_sequencer)) \
      `uvm_fatal("PSEQUENCER", "Cast failed") \
  endfunction

//===----------------------------------------------------------------------===//
// Blocking Task/Function Macros (TLM)
//===----------------------------------------------------------------------===//

`define uvm_blocking_put_imp_decl(SFX) \
  class uvm_blocking_put_imp``SFX #(type T=int, type IMP=int) extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    local IMP m_imp; \
    function new(string name, IMP imp); \
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1); \
      m_imp = imp; \
    endfunction \
    task put(input T t); \
      m_imp.put``SFX(t); \
    endtask \
  endclass

`define uvm_blocking_get_imp_decl(SFX) \
  class uvm_blocking_get_imp``SFX #(type T=int, type IMP=int) extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    local IMP m_imp; \
    function new(string name, IMP imp); \
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1); \
      m_imp = imp; \
    endfunction \
    task get(output T t); \
      m_imp.get``SFX(t); \
    endtask \
  endclass

`define uvm_analysis_imp_decl(SFX) \
  class uvm_analysis_imp``SFX #(type T=int, type IMP=int) extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    local IMP m_imp; \
    function new(string name, IMP imp); \
      super.new(name, imp, UVM_IMPLEMENTATION, 1, 1); \
      m_imp = imp; \
    endfunction \
    function void write(input T t); \
      m_imp.write``SFX(t); \
    endfunction \
  endclass

//===----------------------------------------------------------------------===//
// Configuration Database Macros
//===----------------------------------------------------------------------===//

// These are functions, not macros in real UVM, but defining as macros for stub
`define uvm_config_db_set(CNTXT, INST_NAME, FIELD_NAME, VALUE) \
  uvm_config_db#($type(VALUE))::set(CNTXT, INST_NAME, FIELD_NAME, VALUE)

`define uvm_config_db_get(CNTXT, INST_NAME, FIELD_NAME, VALUE) \
  uvm_config_db#($type(VALUE))::get(CNTXT, INST_NAME, FIELD_NAME, VALUE)

//===----------------------------------------------------------------------===//
// Pack/Unpack Macros
//===----------------------------------------------------------------------===//

`define uvm_pack_int(VAR) \
  __m_uvm_status_container.packer.pack_field_int(VAR, $bits(VAR))

`define uvm_pack_string(VAR) \
  __m_uvm_status_container.packer.pack_string(VAR)

`define uvm_pack_object(VAR) \
  __m_uvm_status_container.packer.pack_object(VAR)

`define uvm_unpack_int(VAR) \
  VAR = __m_uvm_status_container.packer.unpack_field_int($bits(VAR))

`define uvm_unpack_string(VAR) \
  VAR = __m_uvm_status_container.packer.unpack_string()

`define uvm_unpack_object(VAR) \
  __m_uvm_status_container.packer.unpack_object(VAR)

//===----------------------------------------------------------------------===//
// Record Macros
//===----------------------------------------------------------------------===//

`define uvm_record_int(NAME, VALUE, SIZE, RADIX)
`define uvm_record_string(NAME, VALUE)
`define uvm_record_time(NAME, VALUE)
`define uvm_record_real(NAME, VALUE)

//===----------------------------------------------------------------------===//
// Add Macros (for callbacks)
//===----------------------------------------------------------------------===//

`define uvm_add_to_seq_lib(SEQ_TYPE, LIB_TYPE)

`define uvm_set_super_type(TYPE, PARENT)

`endif // UVM_MACROS_SVH
