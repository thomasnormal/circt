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
// Note: Does NOT define get_type_name - use uvm_type_name_decl for that
`define uvm_component_utils(T) \
  typedef uvm_component_registry #(T, `"T`") type_id; \
  static function type_id get_type(); \
    return type_id::get(); \
  endfunction

// UVM_COMPONENT_UTILS_BEGIN/END - For field automation
`define uvm_component_utils_begin(T) \
  `uvm_component_utils(T)

`define uvm_component_utils_end

// UVM_OBJECT_UTILS - Register an object with the factory
// Note: Does NOT define get_type_name - use uvm_type_name_decl for that
`define uvm_object_utils(T) \
  typedef uvm_object_registry #(T, `"T`") type_id; \
  static function type_id get_type(); \
    return type_id::get(); \
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

// uvm_record_int - Record an integral value to a transaction database
// NAME: field name, VALUE: data value, SIZE: bit width, RADIX: display format
`define uvm_record_int(NAME, VALUE, SIZE, RADIX=UVM_NORADIX, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    if (SIZE > 64) \
      RECORDER.record_field(NAME, VALUE, SIZE, RADIX); \
    else \
      RECORDER.record_field_int(NAME, VALUE, SIZE, RADIX); \
  end

// uvm_record_string - Record a string value to a transaction database
`define uvm_record_string(NAME, VALUE, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    RECORDER.record_string(NAME, VALUE); \
  end

// uvm_record_time - Record a time value to a transaction database
`define uvm_record_time(NAME, VALUE, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    RECORDER.record_time(NAME, VALUE); \
  end

// uvm_record_real - Record a real value to a transaction database
`define uvm_record_real(NAME, VALUE, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    RECORDER.record_field_real(NAME, VALUE); \
  end

// uvm_record_field - Record a generic field to a transaction database
`define uvm_record_field(NAME, VALUE, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    RECORDER.record_generic(NAME, $sformatf("%p", VALUE)); \
  end

// uvm_record_enum - Record an enum value to a transaction database
`define uvm_record_enum(NAME, VALUE, TYPE, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    if (VALUE.name() == "") \
      RECORDER.record_generic(NAME, $sformatf("%0d", VALUE), `"TYPE`"); \
    else \
      RECORDER.record_generic(NAME, VALUE.name(), `"TYPE`"); \
  end

// uvm_record_object - Record an object to a transaction database
`define uvm_record_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, RECORDER=recorder) \
  `uvm_record_named_object(`"VALUE`", VALUE, RECURSION_POLICY, RECORDER)

`define uvm_record_named_object(NAME, VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, RECORDER=recorder) \
  if (RECORDER != null && RECORDER.is_open()) begin \
    RECORDER.record_object(NAME, VALUE); \
  end

// uvm_record_attribute - Vendor-independent macro for recording attributes
`define uvm_record_attribute(TR_HANDLE, NAME, VALUE, RECORDER=recorder) \
  RECORDER.record_generic(NAME, $sformatf("%p", VALUE))

// uvm_record_qda_int - Record a queue/dynamic array of integers
`define uvm_record_qda_int(ARG, RADIX, RECORDER=recorder) \
  begin \
    int sz__ = $size(ARG); \
    if (sz__ == 0) begin \
      `uvm_record_int(`"ARG`", 0, 32, UVM_DEC, RECORDER) \
    end \
    else if (sz__ < 10) begin \
      foreach(ARG[i]) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_int(nm__, ARG[i], $bits(ARG[i]), RADIX, RECORDER) \
      end \
    end \
    else begin \
      for (int i=0; i<5; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_int(nm__, ARG[i], $bits(ARG[i]), RADIX, RECORDER) \
      end \
      for (int i=sz__-5; i<sz__; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_int(nm__, ARG[i], $bits(ARG[i]), RADIX, RECORDER) \
      end \
    end \
  end

// uvm_record_qda_string - Record a queue/dynamic array of strings
`define uvm_record_qda_string(ARG, RECORDER=recorder) \
  begin \
    int sz__; \
    foreach (ARG[i]) sz__ = i; \
    if (sz__ == 0) begin \
      `uvm_record_int(`"ARG`", 0, 32, UVM_DEC, RECORDER) \
    end \
    else if (sz__ < 10) begin \
      foreach(ARG[i]) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_string(nm__, ARG[i], RECORDER) \
      end \
    end \
    else begin \
      for (int i=0; i<5; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_string(nm__, ARG[i], RECORDER) \
      end \
      for (int i=sz__-5; i<sz__; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_string(nm__, ARG[i], RECORDER) \
      end \
    end \
  end

// uvm_record_qda_enum - Record a queue/dynamic array of enums
`define uvm_record_qda_enum(ARG, T, RECORDER=recorder) \
  begin \
    int sz__ = $size(ARG); \
    if (sz__ == 0) begin \
      `uvm_record_int(`"ARG`", 0, 32, UVM_DEC, RECORDER) \
    end \
    else if (sz__ < 10) begin \
      foreach(ARG[i]) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_enum(nm__, ARG[i], T, RECORDER) \
      end \
    end \
    else begin \
      for (int i=0; i<5; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_enum(nm__, ARG[i], T, RECORDER) \
      end \
      for (int i=sz__-5; i<sz__; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_enum(nm__, ARG[i], T, RECORDER) \
      end \
    end \
  end

// uvm_record_qda_object - Record a queue/dynamic array of objects
`define uvm_record_qda_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, RECORDER=recorder) \
  begin \
    int sz__ = $size(VALUE); \
    if (sz__ == 0) begin \
      `uvm_record_int(`"VALUE`", 0, 32, UVM_DEC, RECORDER) \
    end \
    else if (sz__ < 10) begin \
      foreach(VALUE[__tmp_index]) begin \
        `uvm_record_named_object($sformatf("%s[%0d]", `"VALUE`", __tmp_index), \
                                 VALUE[__tmp_index], RECURSION_POLICY, RECORDER) \
      end \
    end \
    else begin \
      for (int __tmp_index=0; __tmp_index<5; ++__tmp_index) begin \
        `uvm_record_named_object($sformatf("%s[%0d]", `"VALUE`", __tmp_index), \
                                 VALUE[__tmp_index], RECURSION_POLICY, RECORDER) \
      end \
      for (int __tmp_index=sz__-5; __tmp_index<sz__; ++__tmp_index) begin \
        `uvm_record_named_object($sformatf("%s[%0d]", `"VALUE`", __tmp_index), \
                                 VALUE[__tmp_index], RECURSION_POLICY, RECORDER) \
      end \
    end \
  end

// uvm_record_qda_real - Record a queue/dynamic array of reals
`define uvm_record_qda_real(ARG, RECORDER=recorder) \
  begin \
    int sz__; \
    foreach (ARG[i]) sz__ = i; \
    if (sz__ == 0) begin \
      `uvm_record_int(`"ARG`", 0, 32, UVM_DEC, RECORDER) \
    end \
    else if (sz__ < 10) begin \
      foreach(ARG[i]) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_real(nm__, ARG[i], RECORDER) \
      end \
    end \
    else begin \
      for (int i=0; i<5; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_real(nm__, ARG[i], RECORDER) \
      end \
      for (int i=sz__-5; i<sz__; ++i) begin \
        string nm__ = $sformatf("%s[%0d]", `"ARG`", i); \
        `uvm_record_real(nm__, ARG[i], RECORDER) \
      end \
    end \
  end

//===----------------------------------------------------------------------===//
// Copier Macros
//===----------------------------------------------------------------------===//

// uvm_copier_get_function - Generate a get function for copier state tracking
// FUNCTION: name of the function to generate (e.g., first, prev)
// This macro creates helper functions used by the uvm_copier class for
// tracking copy state during recursive object copying operations.
`define uvm_copier_get_function(FUNCTION) \
  function int get_``FUNCTION``_copy(uvm_object rhs, ref uvm_object lhs); \
    if (m_recur_states.exists(rhs)) \
      return m_recur_states[rhs].FUNCTION(lhs); \
    return 0; \
  endfunction : get_``FUNCTION``_copy

// uvm_copy_object - Copy an object with proper policy handling
`define uvm_copy_object(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COPIER=copier) \
  if (LVALUE != RVALUE) begin \
    if ((RVALUE == null) || \
        (POLICY == UVM_REFERENCE) || \
        ((POLICY == UVM_DEFAULT_POLICY) && \
         (COPIER.get_recursion_policy() == UVM_REFERENCE))) begin \
      LVALUE = RVALUE; \
    end \
    else begin \
      COPIER.copy_object(LVALUE, RVALUE); \
    end \
  end

// uvm_copy_aa_object - Copy an associative array of objects
`define uvm_copy_aa_object(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COPIER=copier) \
  if ((POLICY == UVM_REFERENCE) || !RVALUE.size()) \
    LVALUE = RVALUE; \
  else begin \
    LVALUE.delete(); \
    foreach(RVALUE[i]) \
      `uvm_copy_object(LVALUE[i], RVALUE[i], POLICY, COPIER) \
  end

//===----------------------------------------------------------------------===//
// Comparer Macros
//===----------------------------------------------------------------------===//

// m_uvm_compare_threshold_begin/end - Check if comparison should proceed
`define m_uvm_compare_threshold_begin(COMPARER) \
  if ((!COMPARER.get_threshold() || \
       (COMPARER.get_result() < COMPARER.get_threshold()))) begin \

`define m_uvm_compare_threshold_end \
  end

// m_uvm_compare_begin/end - Compare with threshold check
`define m_uvm_compare_begin(LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_threshold_begin(COMPARER) \
    if ((LVALUE) !== (RVALUE)) begin \

`define m_uvm_compare_end \
    end \
  `m_uvm_compare_threshold_end

// uvm_compare_int - Compare integral values
`define uvm_compare_int(LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `uvm_compare_named_int(`"LVALUE`", LVALUE, RVALUE, RADIX, COMPARER)

`define uvm_compare_named_int(NAME, LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
     if ($bits(LVALUE) <= 64) \
       void'(COMPARER.compare_field_int(NAME, LVALUE, RVALUE, $bits(LVALUE), RADIX)); \
     else \
       void'(COMPARER.compare_field(NAME, LVALUE, RVALUE, $bits(LVALUE), RADIX)); \
  `m_uvm_compare_end

// uvm_compare_enum - Compare enum values
`define uvm_compare_enum(LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `uvm_compare_named_enum(`"LVALUE`", LVALUE, RVALUE, TYPE, COMPARER)

`define uvm_compare_named_enum(NAME, LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
     void'(COMPARER.compare_string(NAME, \
                                   $sformatf("%s'(%s)", `"TYPE`", LVALUE.name()), \
                                   $sformatf("%s'(%s)", `"TYPE`", RVALUE.name()))); \
  `m_uvm_compare_end

// uvm_compare_real - Compare real values
`define uvm_compare_real(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_real(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_real(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_threshold_begin(COMPARER) \
    if ((LVALUE) != (RVALUE)) begin \
      void'(COMPARER.compare_field_real(NAME, LVALUE, RVALUE)); \
    end \
  `m_uvm_compare_threshold_end

// uvm_compare_object - Compare objects with recursion policy
`define uvm_compare_object(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `uvm_compare_named_object(`"LVALUE`", LVALUE, RVALUE, POLICY, COMPARER)

`define uvm_compare_named_object(NAME, LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
     void'(COMPARER.compare_object(NAME, LVALUE, RVALUE)); \
  `m_uvm_compare_end

// uvm_compare_string - Compare string values
`define uvm_compare_string(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_string(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_string(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_threshold_begin(COMPARER) \
    if ((LVALUE) != (RVALUE)) begin \
      void'(COMPARER.compare_string(NAME, LVALUE, RVALUE)); \
    end \
  `m_uvm_compare_threshold_end

// uvm_compare_sarray_int - Compare static array of integers
`define uvm_compare_sarray_int(LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `uvm_compare_named_sarray_int(`"LVALUE`", LVALUE, RVALUE, RADIX, COMPARER)

`define uvm_compare_named_sarray_int(NAME, LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach (LVALUE[i]) begin \
      `uvm_compare_named_int($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], RADIX, COMPARER) \
    end \
  `m_uvm_compare_end

// uvm_compare_qda_int - Compare queue/dynamic array of integers
`define uvm_compare_qda_int(LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `uvm_compare_named_qda_int(`"LVALUE`", LVALUE, RVALUE, RADIX, COMPARER)

`define uvm_compare_named_qda_int(NAME, LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    `uvm_compare_named_int($sformatf("%s.size()", NAME), LVALUE.size(), RVALUE.size(), UVM_DEC, COMPARER) \
    `uvm_compare_named_sarray_int(NAME, LVALUE, RVALUE, RADIX, COMPARER) \
  `m_uvm_compare_end

// uvm_compare_sarray_object - Compare static array of objects
`define uvm_compare_sarray_object(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `uvm_compare_named_sarray_object(`"LVALUE`", LVALUE, RVALUE, POLICY, COMPARER)

`define uvm_compare_named_sarray_object(NAME, LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach (LVALUE[i]) begin \
      `uvm_compare_named_object($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], POLICY, COMPARER) \
    end \
  `m_uvm_compare_end

// uvm_compare_qda_object - Compare queue/dynamic array of objects
`define uvm_compare_qda_object(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `uvm_compare_named_qda_object(`"LVALUE`", LVALUE, RVALUE, POLICY, COMPARER)

`define uvm_compare_named_qda_object(NAME, LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    `uvm_compare_named_int($sformatf("%s.size()", NAME), LVALUE.size(), RVALUE.size(), UVM_DEC, COMPARER) \
    `uvm_compare_named_sarray_object(NAME, LVALUE, RVALUE, POLICY, COMPARER) \
  `m_uvm_compare_end

// uvm_compare_sarray_string - Compare static array of strings
`define uvm_compare_sarray_string(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_sarray_string(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_sarray_string(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach (LVALUE[i]) begin \
      `uvm_compare_named_string($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], COMPARER) \
    end \
  `m_uvm_compare_end

// uvm_compare_qda_string - Compare queue/dynamic array of strings
`define uvm_compare_qda_string(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_qda_string(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_qda_string(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    `uvm_compare_named_int($sformatf("%s.size()", NAME), LVALUE.size(), RVALUE.size(), UVM_DEC, COMPARER) \
    `uvm_compare_named_sarray_string(NAME, LVALUE, RVALUE, COMPARER) \
  `m_uvm_compare_end

// uvm_compare_aa_int_string - Compare associative array (int indexed by string)
`define uvm_compare_aa_int_string(LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `uvm_compare_named_aa_int_string(`"LVALUE`", LVALUE, RVALUE, RADIX, COMPARER)

`define uvm_compare_named_aa_int_string(NAME, LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_int($sformatf("%s[%s]", NAME, i), LVALUE[i], RVALUE[i], RADIX, COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

// uvm_compare_aa_int_int - Compare associative array (int indexed by int)
`define uvm_compare_aa_int_int(LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `uvm_compare_named_aa_int_int(`"LVALUE`", LVALUE, RVALUE, RADIX, COMPARER)

`define uvm_compare_named_aa_int_int(NAME, LVALUE, RVALUE, RADIX, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_int($sformatf("%s[%d]", NAME, i), LVALUE[i], RVALUE[i], RADIX, COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

// uvm_compare_sarray_real - Compare static array of reals
`define uvm_compare_sarray_real(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_sarray_real(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_sarray_real(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_threshold_begin(COMPARER) \
    if ((LVALUE) != (RVALUE)) begin \
      foreach (LVALUE[i]) begin \
        `uvm_compare_named_real($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], COMPARER) \
      end \
    end \
  `m_uvm_compare_threshold_end

// uvm_compare_qda_real - Compare queue/dynamic array of reals
`define uvm_compare_qda_real(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_qda_real(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_qda_real(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_threshold_begin(COMPARER) \
    `uvm_compare_named_real($sformatf("%s.size()", NAME), LVALUE.size(), RVALUE.size(), COMPARER) \
    `uvm_compare_named_sarray_real(NAME, LVALUE, RVALUE, COMPARER) \
  `m_uvm_compare_threshold_end

// uvm_compare_sarray_enum - Compare static array of enums
`define uvm_compare_sarray_enum(LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `uvm_compare_named_sarray_enum(`"LVALUE`", LVALUE, RVALUE, TYPE, COMPARER)

`define uvm_compare_named_sarray_enum(NAME, LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach (LVALUE[i]) begin \
      `uvm_compare_named_enum($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], TYPE, COMPARER) \
    end \
  `m_uvm_compare_end

// uvm_compare_qda_enum - Compare queue/dynamic array of enums
`define uvm_compare_qda_enum(LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `uvm_compare_named_qda_enum(`"LVALUE`", LVALUE, RVALUE, TYPE, COMPARER)

`define uvm_compare_named_qda_enum(NAME, LVALUE, RVALUE, TYPE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    `uvm_compare_named_int($sformatf("%s.size()", NAME), LVALUE.size(), RVALUE.size(), UVM_DEC, COMPARER) \
    `uvm_compare_named_sarray_enum(NAME, LVALUE, RVALUE, TYPE, COMPARER) \
  `m_uvm_compare_end

// uvm_compare_aa_object_string - Compare associative array (object indexed by string)
`define uvm_compare_aa_object_string(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `uvm_compare_named_aa_object_string(`"LVALUE`", LVALUE, RVALUE, POLICY, COMPARER)

`define uvm_compare_named_aa_object_string(NAME, LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_object($sformatf("%s[%s]", NAME, i), LVALUE[i], RVALUE[i], POLICY, COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

// uvm_compare_aa_object_int - Compare associative array (object indexed by int)
`define uvm_compare_aa_object_int(LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `uvm_compare_named_aa_object_int(`"LVALUE`", LVALUE, RVALUE, POLICY, COMPARER)

`define uvm_compare_named_aa_object_int(NAME, LVALUE, RVALUE, POLICY=UVM_DEFAULT_POLICY, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_object($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], POLICY, COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

// uvm_compare_aa_string_int - Compare associative array (string indexed by int)
`define uvm_compare_aa_string_int(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_aa_string_int(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_aa_string_int(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_string($sformatf("%s[%0d]", NAME, i), LVALUE[i], RVALUE[i], COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%0d' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

// uvm_compare_aa_string_string - Compare associative array (string indexed by string)
`define uvm_compare_aa_string_string(LVALUE, RVALUE, COMPARER=comparer) \
  `uvm_compare_named_aa_string_string(`"LVALUE`", LVALUE, RVALUE, COMPARER)

`define uvm_compare_named_aa_string_string(NAME, LVALUE, RVALUE, COMPARER=comparer) \
  `m_uvm_compare_begin(LVALUE, RVALUE, COMPARER) \
    foreach(LVALUE[i]) begin \
      if (!RVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in RHS", NAME, i)); \
      end \
      else begin \
        `uvm_compare_named_string($sformatf("%s[%s]", NAME, i), LVALUE[i], RVALUE[i], COMPARER) \
      end \
    end \
    foreach(RVALUE[i]) begin \
      if(!LVALUE.exists(i)) begin \
        COMPARER.print_msg($sformatf("%s: Key '%s' not in LHS", NAME, i)); \
      end \
    end \
  `m_uvm_compare_end

//===----------------------------------------------------------------------===//
// Packer Macros (Extended)
//===----------------------------------------------------------------------===//

// uvm_packer_array_extension_begin/end - Wrapper for packer array extension
`define uvm_packer_array_extension_begin(PACKER) \
  begin

`define uvm_packer_array_extension_end(PACKER) \
  end

// uvm_pack_intN - Pack an integral variable with explicit size
`define uvm_pack_intN(VAR, SIZE, PACKER=packer) \
  if (SIZE <= 64) begin \
     PACKER.pack_field_int(VAR, SIZE); \
  end \
  else begin \
     PACKER.pack_field(VAR, SIZE); \
  end

// uvm_pack_enumN - Pack an enum with explicit size
`define uvm_pack_enumN(VAR, SIZE, PACKER=packer) \
   `uvm_pack_intN(VAR, SIZE, PACKER)

// uvm_pack_sarrayN - Pack a static array with explicit element size
`define uvm_pack_sarrayN(VAR, SIZE, PACKER=packer) \
  `uvm_packer_array_extension_begin(PACKER) \
    foreach(VAR[index]) begin \
      `uvm_pack_intN(VAR[index], SIZE, PACKER) \
    end \
  `uvm_packer_array_extension_end(PACKER)

// uvm_pack_arrayN - Pack a dynamic array with explicit element size
`define uvm_pack_arrayN(VAR, SIZE, PACKER=packer) \
  begin \
    `uvm_pack_intN(VAR.size(), 32, PACKER) \
    `uvm_pack_sarrayN(VAR, SIZE, PACKER) \
  end

// uvm_pack_queueN - Pack a queue with explicit element size
`define uvm_pack_queueN(VAR, SIZE, PACKER=packer) \
   `uvm_pack_arrayN(VAR, SIZE, PACKER)

// uvm_pack_enum - Pack an enum (auto-size)
`define uvm_pack_enum(VAR, PACKER=packer) \
   `uvm_pack_enumN(VAR, $bits(VAR), PACKER)

// uvm_pack_sarray - Pack a static array (auto-size)
`define uvm_pack_sarray(VAR, PACKER=packer) \
   `uvm_pack_sarrayN(VAR, $bits(VAR[0]), PACKER)

// uvm_pack_array - Pack a dynamic array (auto-size)
`define uvm_pack_array(VAR, PACKER=packer) \
   `uvm_pack_arrayN(VAR, $bits(VAR[0]), PACKER)

`define uvm_pack_da(VAR, PACKER=packer) \
  `uvm_pack_array(VAR, PACKER)

// uvm_pack_queue - Pack a queue (auto-size)
`define uvm_pack_queue(VAR, PACKER=packer) \
   `uvm_pack_queueN(VAR, $bits(VAR[0]), PACKER)

// uvm_pack_real - Pack a real value
`define uvm_pack_real(VAR, PACKER=packer) \
  PACKER.pack_real(VAR);

// uvm_unpack_intN - Unpack an integral variable with explicit size
`define uvm_unpack_intN(VAR, SIZE, PACKER=packer) \
  if (SIZE <= 64) begin \
    VAR = PACKER.unpack_field_int(SIZE); \
  end \
  else begin \
    VAR = PACKER.unpack_field(SIZE); \
  end

// uvm_unpack_enumN - Unpack an enum with explicit size
`define uvm_unpack_enumN(VAR, SIZE, TYPE, PACKER=packer) \
  begin \
    VAR = TYPE'(PACKER.unpack_field_int(SIZE)); \
  end

// uvm_unpack_sarrayN - Unpack a static array with explicit element size
`define uvm_unpack_sarrayN(VAR, SIZE, PACKER=packer) \
  `uvm_packer_array_extension_begin(PACKER) \
    foreach (VAR[i]) begin \
      `uvm_unpack_intN(VAR[i], SIZE, PACKER) \
    end \
  `uvm_packer_array_extension_end(PACKER)

// uvm_unpack_arrayN - Unpack a dynamic array with explicit element size
`define uvm_unpack_arrayN(VAR, SIZE, PACKER=packer) \
  begin \
    int sz__; \
    `uvm_unpack_intN(sz__, 32, PACKER) \
    VAR = new[sz__]; \
    `uvm_unpack_sarrayN(VAR, SIZE, PACKER) \
  end

// uvm_unpack_queueN - Unpack a queue with explicit element size
`define uvm_unpack_queueN(VAR, SIZE, PACKER=packer) \
  begin \
    int sz__; \
    `uvm_unpack_intN(sz__, 32, PACKER) \
    while (VAR.size() > sz__) \
      void'(VAR.pop_back()); \
    for (int i=0; i<sz__; i++) \
      `uvm_unpack_intN(VAR[i], SIZE, PACKER) \
  end

// uvm_unpack_enum - Unpack an enum (auto-size)
`define uvm_unpack_enum(VAR, TYPE, PACKER=packer) \
   `uvm_unpack_enumN(VAR, $bits(VAR), TYPE, PACKER)

// uvm_unpack_sarray - Unpack a static array (auto-size)
`define uvm_unpack_sarray(VAR, PACKER=packer) \
   `uvm_unpack_sarrayN(VAR, $bits(VAR[0]), PACKER)

// uvm_unpack_array - Unpack a dynamic array (auto-size)
`define uvm_unpack_array(VAR, PACKER=packer) \
   `uvm_unpack_arrayN(VAR, $bits(VAR[0]), PACKER)

`define uvm_unpack_da(VAR, PACKER=packer) \
  `uvm_unpack_array(VAR, PACKER)

// uvm_unpack_queue - Unpack a queue (auto-size)
`define uvm_unpack_queue(VAR, PACKER=packer) \
   `uvm_unpack_queueN(VAR, $bits(VAR[0]), PACKER)

// uvm_unpack_real - Unpack a real value
`define uvm_unpack_real(VAR, PACKER=packer) \
  VAR = PACKER.unpack_real();

// Associative array pack/unpack macros
`define uvm_pack_aa_int_intN(VAR, SIZE, PACKER=packer) \
  begin \
    `uvm_pack_intN(VAR.num(), 32, PACKER) \
    if (VAR.num()) begin \
      `uvm_pack_intN(SIZE, 32, PACKER) \
      foreach(VAR[i]) begin \
        `uvm_pack_intN(i, SIZE, PACKER) \
        `uvm_pack_int(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_int_intN(VAR, SIZE, PACKER=packer) \
  begin \
    int __num__; \
    `uvm_unpack_intN(__num__, 32, PACKER) \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      bit[SIZE-1:0] __index__; \
      int __sz__; \
      `uvm_unpack_intN(__sz__, 32, PACKER) \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_intN(__index__, SIZE, PACKER) \
        `uvm_unpack_int(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_object_string(VAR, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_string(i, PACKER) \
        `uvm_pack_object(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_object_string(VAR, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      string __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_string(__index__, PACKER) \
        `uvm_unpack_object(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_int_string(VAR, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_string(i, PACKER) \
        `uvm_pack_int(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_int_string(VAR, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      string __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_string(__index__, PACKER) \
        `uvm_unpack_int(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_string_string(VAR, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_string(i, PACKER) \
        `uvm_pack_string(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_string_string(VAR, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      string __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_string(__index__, PACKER) \
        `uvm_unpack_string(VAR[__index__], PACKER) \
      end \
    end \
  end

// Pack/unpack for enum-indexed associative arrays
`define uvm_pack_aa_int_enum(VAR, TYPE, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_enum(i, PACKER) \
        `uvm_pack_int(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_int_enum(VAR, TYPE, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      TYPE __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_enum(__index__, TYPE, PACKER) \
        `uvm_unpack_int(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_object_enum(VAR, TYPE, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_enum(i, PACKER) \
        `uvm_pack_object(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_object_enum(VAR, TYPE, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      TYPE __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_enum(__index__, TYPE, PACKER) \
        `uvm_unpack_object(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_string_enum(VAR, TYPE, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      foreach(VAR[i]) begin \
        `uvm_pack_enum(i, PACKER) \
        `uvm_pack_string(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_string_enum(VAR, TYPE, PACKER=packer) \
  begin \
    int __num__ = PACKER.unpack_field_int(32); \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      TYPE __index__; \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_enum(__index__, TYPE, PACKER) \
        `uvm_unpack_string(VAR[__index__], PACKER) \
      end \
    end \
  end

// Pack/unpack for int-indexed object and string associative arrays
`define uvm_pack_aa_object_intN(VAR, SIZE, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      PACKER.pack_field_int(SIZE, 32); \
      foreach(VAR[i]) begin \
        `uvm_pack_intN(i, SIZE, PACKER) \
        `uvm_pack_object(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_object_intN(VAR, SIZE, PACKER=packer) \
  begin \
    int __num__; \
    `uvm_unpack_intN(__num__, 32, PACKER) \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      bit[SIZE-1:0] __index__; \
      int __sz__; \
      `uvm_unpack_intN(__sz__, 32, PACKER) \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_intN(__index__, __sz__, PACKER) \
        `uvm_unpack_object(VAR[__index__], PACKER) \
      end \
    end \
  end

`define uvm_pack_aa_string_intN(VAR, SIZE, PACKER=packer) \
  begin \
    PACKER.pack_field_int(VAR.num(), 32); \
    if (VAR.num()) begin \
      PACKER.pack_field_int(SIZE, 32); \
      foreach(VAR[i]) begin \
        `uvm_pack_intN(i, SIZE, PACKER) \
        `uvm_pack_string(VAR[i], PACKER) \
      end \
    end \
  end

`define uvm_unpack_aa_string_intN(VAR, SIZE, PACKER=packer) \
  begin \
    int __num__; \
    `uvm_unpack_intN(__num__, 32, PACKER) \
    if (__num__ == 0) \
      VAR.delete(); \
    else begin \
      bit[SIZE-1:0] __index__; \
      int __sz__; \
      `uvm_unpack_intN(__sz__, 32, PACKER) \
      for (int __i__ = 0; __i__ < __num__; __i__++) begin \
        `uvm_unpack_intN(__index__, __sz__, PACKER) \
        `uvm_unpack_string(VAR[__index__], PACKER) \
      end \
    end \
  end

// Pack/unpack for real arrays
`define uvm_pack_sarray_real(VAR, PACKER=packer) \
  `uvm_packer_array_extension_begin(PACKER) \
    foreach(VAR[index]) \
      `uvm_pack_real(VAR[index], PACKER) \
  `uvm_packer_array_extension_end(PACKER)

`define m_uvm_pack_qda_real(VAR, PACKER=packer) \
  `uvm_pack_intN(VAR.size(), 32, PACKER) \
  `uvm_pack_sarray_real(VAR, PACKER)

`define uvm_pack_queue_real(VAR, PACKER=packer) \
  `m_uvm_pack_qda_real(VAR, PACKER)

`define uvm_pack_da_real(VAR, PACKER=packer) \
  `m_uvm_pack_qda_real(VAR, PACKER)

`define uvm_unpack_sarray_real(VAR, PACKER=packer) \
  `uvm_packer_array_extension_begin(PACKER) \
    foreach(VAR[index]) \
      `uvm_unpack_real(VAR[index], PACKER) \
  `uvm_packer_array_extension_end(PACKER)

`define uvm_unpack_da_real(VAR, PACKER=packer) \
  begin \
    int tmp_size__; \
    `uvm_unpack_intN(tmp_size__, 32, PACKER) \
    VAR = new [tmp_size__]; \
    `uvm_unpack_sarray_real(VAR, PACKER) \
  end

`define uvm_unpack_queue_real(VAR, PACKER=packer) \
  begin \
    int tmp_size__; \
    `uvm_unpack_intN(tmp_size__, 32, PACKER) \
    while (VAR.size() > tmp_size__) \
      void'(VAR.pop_back()); \
    for (int i = 0; i < tmp_size__; i++) \
      `uvm_unpack_real(VAR[i], PACKER) \
  end

//===----------------------------------------------------------------------===//
// Add Macros (for callbacks)
//===----------------------------------------------------------------------===//

`define uvm_add_to_seq_lib(SEQ_TYPE, LIB_TYPE) \
  static bit add_``SEQ_TYPE``_to_seq_lib_``LIB_TYPE = \
    LIB_TYPE::m_add_typewide_sequence(SEQ_TYPE::get_type());

`define uvm_set_super_type(TYPE, PARENT)

//===----------------------------------------------------------------------===//
// Sequence Library Macros
//===----------------------------------------------------------------------===//

`define uvm_sequence_library_utils(TYPE) \
  static protected uvm_object_wrapper m_typewide_sequences[$]; \
  function void init_sequence_library(); \
    foreach (TYPE::m_typewide_sequences[i]) \
      sequences.push_back(TYPE::m_typewide_sequences[i]); \
  endfunction \
  static function void add_typewide_sequence(uvm_object_wrapper seq_type); \
    if (m_static_check(seq_type)) \
      TYPE::m_typewide_sequences.push_back(seq_type); \
  endfunction \
  static function void add_typewide_sequences(uvm_object_wrapper seq_types[$]); \
    foreach (seq_types[i]) \
      TYPE::add_typewide_sequence(seq_types[i]); \
  endfunction \
  static function bit m_add_typewide_sequence(uvm_object_wrapper seq_type); \
    TYPE::add_typewide_sequence(seq_type); \
    return 1; \
  endfunction

//===----------------------------------------------------------------------===//
// Type Name and String Macros
//===----------------------------------------------------------------------===//

// UVM_STRING_QUEUE_STREAMING_PACK - Join string queue elements
`ifndef UVM_STRING_QUEUE_STREAMING_PACK
  `define UVM_STRING_QUEUE_STREAMING_PACK(q) m_uvm_string_queue_join(q)
`endif

// uvm_typename - Get type name as string
`define uvm_typename(T) `"T`"

// uvm_type_name_decl - Declare get_type_name() method
`define uvm_type_name_decl(TNAME_STRING) \
  virtual function string get_type_name(); \
    return TNAME_STRING; \
  endfunction

//===----------------------------------------------------------------------===//
// Abstract Object Utilities
//===----------------------------------------------------------------------===//

`define uvm_object_abstract_utils(T) \
  static function string type_name(); \
    return `"T`"; \
  endfunction

`define uvm_object_abstract_param_utils(T) \
  `uvm_object_abstract_utils(T)

`define uvm_object_abstract_utils_begin(T) \
  `uvm_object_abstract_utils(T)

`define uvm_object_abstract_utils_end

//===----------------------------------------------------------------------===//
// Component Abstract Utilities
//===----------------------------------------------------------------------===//

`define uvm_component_abstract_utils(T) \
  static function string type_name(); \
    return `"T`"; \
  endfunction

//===----------------------------------------------------------------------===//
// Additional Global Defines
//===----------------------------------------------------------------------===//

`ifndef UVM_MAX_STREAMBITS
  `define UVM_MAX_STREAMBITS 4096
`endif

`ifndef UVM_FIELD_FLAG_SIZE
  `define UVM_FIELD_FLAG_SIZE 64
`endif

`ifndef UVM_LINE_WIDTH
  `define UVM_LINE_WIDTH 120
`endif

`ifndef UVM_NUM_LINES
  `define UVM_NUM_LINES 100
`endif

`ifndef UVM_FIELD_FLAG_RESERVED_BITS
  `define UVM_FIELD_FLAG_RESERVED_BITS 28
`endif

`ifndef UVM_PACKER_MAX_BYTES
  `define UVM_PACKER_MAX_BYTES 4096
`endif

//===----------------------------------------------------------------------===//
// TLM Implementation Port Declaration Macros
//===----------------------------------------------------------------------===//

// TLM Mask definitions
`define UVM_TLM_BLOCKING_PUT_MASK          (1<<0)
`define UVM_TLM_BLOCKING_GET_MASK          (1<<1)
`define UVM_TLM_BLOCKING_PEEK_MASK         (1<<2)
`define UVM_TLM_BLOCKING_TRANSPORT_MASK    (1<<3)
`define UVM_TLM_NONBLOCKING_PUT_MASK       (1<<4)
`define UVM_TLM_NONBLOCKING_GET_MASK       (1<<5)
`define UVM_TLM_NONBLOCKING_PEEK_MASK      (1<<6)
`define UVM_TLM_NONBLOCKING_TRANSPORT_MASK (1<<7)
`define UVM_TLM_ANALYSIS_MASK              (1<<8)
`define UVM_TLM_MASTER_BIT_MASK            (1<<9)
`define UVM_TLM_SLAVE_BIT_MASK             (1<<10)

// Combination TLM masks
`define UVM_TLM_PUT_MASK                  (`UVM_TLM_BLOCKING_PUT_MASK | `UVM_TLM_NONBLOCKING_PUT_MASK)
`define UVM_TLM_GET_MASK                  (`UVM_TLM_BLOCKING_GET_MASK | `UVM_TLM_NONBLOCKING_GET_MASK)
`define UVM_TLM_PEEK_MASK                 (`UVM_TLM_BLOCKING_PEEK_MASK | `UVM_TLM_NONBLOCKING_PEEK_MASK)
`define UVM_TLM_BLOCKING_GET_PEEK_MASK    (`UVM_TLM_BLOCKING_GET_MASK | `UVM_TLM_BLOCKING_PEEK_MASK)
`define UVM_TLM_BLOCKING_MASTER_MASK      (`UVM_TLM_BLOCKING_PUT_MASK | `UVM_TLM_BLOCKING_GET_MASK | `UVM_TLM_BLOCKING_PEEK_MASK | `UVM_TLM_MASTER_BIT_MASK)
`define UVM_TLM_BLOCKING_SLAVE_MASK       (`UVM_TLM_BLOCKING_PUT_MASK | `UVM_TLM_BLOCKING_GET_MASK | `UVM_TLM_BLOCKING_PEEK_MASK | `UVM_TLM_SLAVE_BIT_MASK)
`define UVM_TLM_NONBLOCKING_GET_PEEK_MASK (`UVM_TLM_NONBLOCKING_GET_MASK | `UVM_TLM_NONBLOCKING_PEEK_MASK)
`define UVM_TLM_NONBLOCKING_MASTER_MASK   (`UVM_TLM_NONBLOCKING_PUT_MASK | `UVM_TLM_NONBLOCKING_GET_MASK | `UVM_TLM_NONBLOCKING_PEEK_MASK | `UVM_TLM_MASTER_BIT_MASK)
`define UVM_TLM_NONBLOCKING_SLAVE_MASK    (`UVM_TLM_NONBLOCKING_PUT_MASK | `UVM_TLM_NONBLOCKING_GET_MASK | `UVM_TLM_NONBLOCKING_PEEK_MASK | `UVM_TLM_SLAVE_BIT_MASK)
`define UVM_TLM_GET_PEEK_MASK             (`UVM_TLM_GET_MASK | `UVM_TLM_PEEK_MASK)
`define UVM_TLM_MASTER_MASK               (`UVM_TLM_BLOCKING_MASTER_MASK | `UVM_TLM_NONBLOCKING_MASTER_MASK)
`define UVM_TLM_SLAVE_MASK                (`UVM_TLM_BLOCKING_SLAVE_MASK | `UVM_TLM_NONBLOCKING_SLAVE_MASK)
`define UVM_TLM_TRANSPORT_MASK            (`UVM_TLM_BLOCKING_TRANSPORT_MASK | `UVM_TLM_NONBLOCKING_TRANSPORT_MASK)

// Sequence item masks
`define UVM_SEQ_ITEM_GET_NEXT_ITEM_MASK       (1<<0)
`define UVM_SEQ_ITEM_TRY_NEXT_ITEM_MASK       (1<<1)
`define UVM_SEQ_ITEM_ITEM_DONE_MASK           (1<<2)
`define UVM_SEQ_ITEM_HAS_DO_AVAILABLE_MASK    (1<<3)
`define UVM_SEQ_ITEM_WAIT_FOR_SEQUENCES_MASK  (1<<4)
`define UVM_SEQ_ITEM_PUT_RESPONSE_MASK        (1<<5)
`define UVM_SEQ_ITEM_PUT_MASK                 (1<<6)
`define UVM_SEQ_ITEM_GET_MASK                 (1<<7)
`define UVM_SEQ_ITEM_PEEK_MASK                (1<<8)
`define UVM_SEQ_ITEM_PULL_MASK  (`UVM_SEQ_ITEM_GET_NEXT_ITEM_MASK | `UVM_SEQ_ITEM_TRY_NEXT_ITEM_MASK | \
                        `UVM_SEQ_ITEM_ITEM_DONE_MASK | `UVM_SEQ_ITEM_HAS_DO_AVAILABLE_MASK | \
                        `UVM_SEQ_ITEM_WAIT_FOR_SEQUENCES_MASK | `UVM_SEQ_ITEM_PUT_RESPONSE_MASK | \
                        `UVM_SEQ_ITEM_PUT_MASK | `UVM_SEQ_ITEM_GET_MASK | `UVM_SEQ_ITEM_PEEK_MASK)
`define UVM_SEQ_ITEM_UNI_PULL_MASK (`UVM_SEQ_ITEM_GET_NEXT_ITEM_MASK | `UVM_SEQ_ITEM_TRY_NEXT_ITEM_MASK | \
                           `UVM_SEQ_ITEM_ITEM_DONE_MASK | `UVM_SEQ_ITEM_HAS_DO_AVAILABLE_MASK | \
                           `UVM_SEQ_ITEM_WAIT_FOR_SEQUENCES_MASK | `UVM_SEQ_ITEM_GET_MASK | \
                           `UVM_SEQ_ITEM_PEEK_MASK)
`define UVM_SEQ_ITEM_PUSH_MASK  (`UVM_SEQ_ITEM_PUT_MASK)

// TLM IMP common macro (stub)
`define UVM_IMP_COMMON(MASK, TYPE_NAME, IMP) \
  local IMP m_imp; \
  function new(string name, IMP imp); \
    super.new(name, imp, UVM_IMPLEMENTATION, 1, 1); \
    m_imp = imp; \
  endfunction

`define UVM_MS_IMP_COMMON(MASK, TYPE_NAME) \
  local this_req_type m_req_imp; \
  local this_rsp_type m_rsp_imp; \
  function new(string name, this_imp_type imp, this_req_type req_imp = null, this_rsp_type rsp_imp = null); \
    super.new(name, imp, UVM_IMPLEMENTATION, 1, 1); \
    if (req_imp == null) $cast(m_req_imp, imp); \
    else m_req_imp = req_imp; \
    if (rsp_imp == null) $cast(m_rsp_imp, imp); \
    else m_rsp_imp = rsp_imp; \
  endfunction

// TLM implementation macros with suffix support
`define UVM_BLOCKING_PUT_IMP_SFX(SFX, imp, TYPE, arg) \
  task put(input TYPE arg); imp.put``SFX(arg); endtask

`define UVM_BLOCKING_GET_IMP_SFX(SFX, imp, TYPE, arg) \
  task get(output TYPE arg); imp.get``SFX(arg); endtask

`define UVM_BLOCKING_PEEK_IMP_SFX(SFX, imp, TYPE, arg) \
  task peek(output TYPE arg); imp.peek``SFX(arg); endtask

`define UVM_NONBLOCKING_PUT_IMP_SFX(SFX, imp, TYPE, arg) \
  function bit try_put(input TYPE arg); \
    if (!imp.try_put``SFX(arg)) return 0; \
    return 1; \
  endfunction \
  function bit can_put(); return imp.can_put``SFX(); endfunction

`define UVM_NONBLOCKING_GET_IMP_SFX(SFX, imp, TYPE, arg) \
  function bit try_get(output TYPE arg); \
    if (!imp.try_get``SFX(arg)) return 0; \
    return 1; \
  endfunction \
  function bit can_get(); return imp.can_get``SFX(); endfunction

`define UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, imp, TYPE, arg) \
  function bit try_peek(output TYPE arg); \
    if (!imp.try_peek``SFX(arg)) return 0; \
    return 1; \
  endfunction \
  function bit can_peek(); return imp.can_peek``SFX(); endfunction

`define UVM_BLOCKING_TRANSPORT_IMP_SFX(SFX, imp, REQ, RSP, req_arg, rsp_arg) \
  task transport(input REQ req_arg, output RSP rsp_arg); \
    imp.transport``SFX(req_arg, rsp_arg); \
  endtask

`define UVM_NONBLOCKING_TRANSPORT_IMP_SFX(SFX, imp, REQ, RSP, req_arg, rsp_arg) \
  function bit nb_transport(input REQ req_arg, output RSP rsp_arg); \
    if (imp) return imp.nb_transport``SFX(req_arg, rsp_arg); \
  endfunction

// Full TLM port declaration macros
`define uvm_put_imp_decl(SFX) \
  class uvm_put_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_PUT_MASK, `"uvm_put_imp``SFX`", IMP) \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_get_imp_decl(SFX) \
  class uvm_get_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_GET_MASK, `"uvm_get_imp``SFX`", IMP) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_peek_imp_decl(SFX) \
  class uvm_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_PEEK_MASK, `"uvm_peek_imp``SFX`", IMP) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_blocking_get_peek_imp_decl(SFX) \
  class uvm_blocking_get_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_BLOCKING_GET_PEEK_MASK, `"uvm_blocking_get_peek_imp``SFX`", IMP) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_nonblocking_get_peek_imp_decl(SFX) \
  class uvm_nonblocking_get_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_NONBLOCKING_GET_PEEK_MASK, `"uvm_nonblocking_get_peek_imp``SFX`", IMP) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_get_peek_imp_decl(SFX) \
  class uvm_get_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_GET_PEEK_MASK, `"uvm_get_peek_imp``SFX`", IMP) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_nonblocking_put_imp_decl(SFX) \
  class uvm_nonblocking_put_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_NONBLOCKING_PUT_MASK, `"uvm_nonblocking_put_imp``SFX`", IMP) \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_nonblocking_get_imp_decl(SFX) \
  class uvm_nonblocking_get_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_NONBLOCKING_GET_MASK, `"uvm_nonblocking_get_imp``SFX`", IMP) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_nonblocking_peek_imp_decl(SFX) \
  class uvm_nonblocking_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_NONBLOCKING_PEEK_MASK, `"uvm_nonblocking_peek_imp``SFX`", IMP) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_blocking_transport_imp_decl(SFX) \
  class uvm_blocking_transport_imp``SFX #(type REQ=int, type RSP=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    `UVM_IMP_COMMON(`UVM_TLM_BLOCKING_TRANSPORT_MASK, `"uvm_blocking_transport_imp``SFX`", IMP) \
    `UVM_BLOCKING_TRANSPORT_IMP_SFX(SFX, m_imp, REQ, RSP, req, rsp) \
  endclass

`define uvm_nonblocking_transport_imp_decl(SFX) \
  class uvm_nonblocking_transport_imp``SFX #(type REQ=int, type RSP=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    `UVM_IMP_COMMON(`UVM_TLM_NONBLOCKING_TRANSPORT_MASK, `"uvm_nonblocking_transport_imp``SFX`", IMP) \
    `UVM_NONBLOCKING_TRANSPORT_IMP_SFX(SFX, m_imp, REQ, RSP, req, rsp) \
  endclass

`define uvm_transport_imp_decl(SFX) \
  class uvm_transport_imp``SFX #(type REQ=int, type RSP=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    `UVM_IMP_COMMON(`UVM_TLM_TRANSPORT_MASK, `"uvm_transport_imp``SFX`", IMP) \
    `UVM_BLOCKING_TRANSPORT_IMP_SFX(SFX, m_imp, REQ, RSP, req, rsp) \
    `UVM_NONBLOCKING_TRANSPORT_IMP_SFX(SFX, m_imp, REQ, RSP, req, rsp) \
  endclass

// Blocking-only TLM port declaration macros
`define uvm_blocking_put_imp_decl(SFX) \
  class uvm_blocking_put_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_BLOCKING_PUT_MASK, `"uvm_blocking_put_imp``SFX`", IMP) \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_blocking_get_imp_decl(SFX) \
  class uvm_blocking_get_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_BLOCKING_GET_MASK, `"uvm_blocking_get_imp``SFX`", IMP) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_imp, T, t) \
  endclass

`define uvm_blocking_peek_imp_decl(SFX) \
  class uvm_blocking_peek_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_BLOCKING_PEEK_MASK, `"uvm_blocking_peek_imp``SFX`", IMP) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_imp, T, t) \
  endclass

// Analysis imp declaration macro
`define uvm_analysis_imp_decl(SFX) \
  class uvm_analysis_imp``SFX #(type T=int, type IMP=int) \
    extends uvm_port_base #(uvm_tlm_if_base #(T,T)); \
    `UVM_IMP_COMMON(`UVM_TLM_ANALYSIS_MASK, `"uvm_analysis_imp``SFX`", IMP) \
    function void write(input T t); \
      m_imp.write``SFX(t); \
    endfunction \
  endclass

// Master/Slave TLM port declaration macros
`define uvm_blocking_master_imp_decl(SFX) \
  class uvm_blocking_master_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                                       type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_BLOCKING_MASTER_MASK, `"uvm_blocking_master_imp``SFX`") \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
  endclass

`define uvm_nonblocking_master_imp_decl(SFX) \
  class uvm_nonblocking_master_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                                          type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_NONBLOCKING_MASTER_MASK, `"uvm_nonblocking_master_imp``SFX`") \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
  endclass

`define uvm_master_imp_decl(SFX) \
  class uvm_master_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                              type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(REQ, RSP)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_MASTER_MASK, `"uvm_master_imp``SFX`") \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
  endclass

`define uvm_blocking_slave_imp_decl(SFX) \
  class uvm_blocking_slave_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                                      type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_BLOCKING_SLAVE_MASK, `"uvm_blocking_slave_imp``SFX`") \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_req_imp, REQ, t) \
  endclass

`define uvm_nonblocking_slave_imp_decl(SFX) \
  class uvm_nonblocking_slave_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                                         type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_NONBLOCKING_SLAVE_MASK, `"uvm_nonblocking_slave_imp``SFX`") \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_req_imp, REQ, t) \
  endclass

`define uvm_slave_imp_decl(SFX) \
  class uvm_slave_imp``SFX #(type REQ=int, type RSP=int, type IMP=int, \
                             type REQ_IMP=IMP, type RSP_IMP=IMP) \
    extends uvm_port_base #(uvm_tlm_if_base #(RSP, REQ)); \
    typedef IMP     this_imp_type; \
    typedef REQ_IMP this_req_type; \
    typedef RSP_IMP this_rsp_type; \
    `UVM_MS_IMP_COMMON(`UVM_TLM_SLAVE_MASK, `"uvm_slave_imp``SFX`") \
    `UVM_BLOCKING_PUT_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_NONBLOCKING_PUT_IMP_SFX(SFX, m_rsp_imp, RSP, t) \
    `UVM_BLOCKING_GET_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_BLOCKING_PEEK_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_NONBLOCKING_GET_IMP_SFX(SFX, m_req_imp, REQ, t) \
    `UVM_NONBLOCKING_PEEK_IMP_SFX(SFX, m_req_imp, REQ, t) \
  endclass

// Sequence item pull IMP macro
`define UVM_SEQ_ITEM_PULL_IMP(imp, REQ, RSP, req_arg, rsp_arg) \
  function void disable_auto_item_recording(); imp.disable_auto_item_recording(); endfunction \
  function bit is_auto_item_recording_enabled(); return imp.is_auto_item_recording_enabled(); endfunction \
  task get_next_item(output REQ req_arg); imp.get_next_item(req_arg); endtask \
  task try_next_item(output REQ req_arg); imp.try_next_item(req_arg); endtask \
  function void item_done(input RSP rsp_arg = null); imp.item_done(rsp_arg); endfunction \
  task wait_for_sequences(); imp.wait_for_sequences(); endtask \
  function bit has_do_available(); return imp.has_do_available(); endfunction \
  function void put_response(input RSP rsp_arg); imp.put_response(rsp_arg); endfunction \
  task get(output REQ req_arg); imp.get(req_arg); endtask \
  task peek(output REQ req_arg); imp.peek(req_arg); endtask \
  task put(input RSP rsp_arg); imp.put(rsp_arg); endtask

//===----------------------------------------------------------------------===//
// Printer Macros
//===----------------------------------------------------------------------===//

`define uvm_print_int(VALUE, SIZE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  `uvm_print_named_int(`"VALUE`", VALUE, SIZE, RADIX, VALUE_TYPE, PRINTER)

`define uvm_print_named_int(NAME, VALUE, SIZE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  if (SIZE > 64) \
    PRINTER.print_field(NAME, VALUE, SIZE, RADIX, ".", `"VALUE_TYPE`"); \
  else \
    PRINTER.print_field_int(NAME, VALUE, SIZE, RADIX, ".", `"VALUE_TYPE`");

`define uvm_print_real(VALUE, PRINTER=printer) \
  `uvm_print_named_real(`"VALUE`", VALUE, PRINTER)

`define uvm_print_named_real(NAME, VALUE, PRINTER=printer) \
  PRINTER.print_real(NAME, VALUE);

`define uvm_print_enum(TYPE, VALUE, PRINTER=printer) \
  `uvm_print_named_enum(TYPE, `"VALUE`", VALUE, PRINTER)

`define uvm_print_named_enum(TYPE, NAME, VALUE, PRINTER=printer) \
  if (VALUE.name() == "") \
    `uvm_print_named_int(NAME, VALUE, $bits(VALUE), UVM_NORADIX, TYPE, PRINTER) \
  else \
    PRINTER.print_generic(NAME, `"TYPE`", $bits(VALUE), VALUE.name());

`define uvm_print_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  `uvm_print_named_object(`"VALUE`", VALUE, RECURSION_POLICY, PRINTER)

`define uvm_print_named_object(NAME, VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  PRINTER.print_object(NAME, VALUE);

`define uvm_print_string(VALUE, PRINTER=printer) \
  `uvm_print_named_string(`"VALUE`", VALUE, PRINTER)

`define uvm_print_named_string(NAME, VALUE, PRINTER=printer) \
  PRINTER.print_string(NAME, VALUE);

`define uvm_print_array_int(VALUE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "da(integral)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_int($sformatf("[%0d]", i), VALUE[i], $bits(VALUE[i]), RADIX, VALUE_TYPE, PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_sarray_int(VALUE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", $size(VALUE), "sa(integral)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_int($sformatf("[%0d]", i), VALUE[i], $bits(VALUE[i]), RADIX, VALUE_TYPE, PRINTER) \
    PRINTER.print_array_footer($size(VALUE)); \
  end

`define uvm_print_queue_int(VALUE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "queue(integral)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_int($sformatf("[%0d]", i), VALUE[i], $bits(VALUE[i]), RADIX, VALUE_TYPE, PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_array_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "da(object)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_object($sformatf("[%0d]", i), VALUE[i], RECURSION_POLICY, PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_sarray_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", $size(VALUE), "sa(object)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_object($sformatf("[%0d]", i), VALUE[i], RECURSION_POLICY, PRINTER) \
    PRINTER.print_array_footer($size(VALUE)); \
  end

`define uvm_print_queue_object(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "queue(object)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_object($sformatf("[%0d]", i), VALUE[i], RECURSION_POLICY, PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_array_string(VALUE, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "da(string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_string($sformatf("[%0d]", i), VALUE[i], PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_sarray_string(VALUE, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", $size(VALUE), "sa(string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_string($sformatf("[%0d]", i), VALUE[i], PRINTER) \
    PRINTER.print_array_footer($size(VALUE)); \
  end

`define uvm_print_queue_string(VALUE, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.size(), "queue(string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_string($sformatf("[%0d]", i), VALUE[i], PRINTER) \
    PRINTER.print_array_footer(VALUE.size()); \
  end

`define uvm_print_aa_int_string(VALUE, RADIX=UVM_NORADIX, VALUE_TYPE=integral, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(integral,string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_int($sformatf("[%s]", i), VALUE[i], $bits(VALUE[i]), RADIX, VALUE_TYPE, PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

`define uvm_print_aa_object_string(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(object,string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_object($sformatf("[%s]", i), VALUE[i], RECURSION_POLICY, PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

`define uvm_print_aa_string_string(VALUE, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(string,string)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_string($sformatf("[%s]", i), VALUE[i], PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

`define uvm_print_aa_int_int(VALUE, RADIX=UVM_NORADIX, VALUE_TYPE=int, INDEX_TYPE=int, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(integral,int)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_int($sformatf("[%0d]", i), VALUE[i], $bits(VALUE[i]), RADIX, VALUE_TYPE, PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

`define uvm_print_aa_object_int(VALUE, RECURSION_POLICY=UVM_DEFAULT_POLICY, INDEX_TYPE=int, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(object,int)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_object($sformatf("[%0d]", i), VALUE[i], RECURSION_POLICY, PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

`define uvm_print_aa_string_int(VALUE, INDEX_TYPE=int, PRINTER=printer) \
  begin \
    PRINTER.print_array_header(`"VALUE`", VALUE.num(), "aa(string,int)"); \
    foreach(VALUE[i]) \
      `uvm_print_named_string($sformatf("[%0d]", i), VALUE[i], PRINTER) \
    PRINTER.print_array_footer(VALUE.num()); \
  end

//===----------------------------------------------------------------------===//
// Message Context Macros
//===----------------------------------------------------------------------===//

`define uvm_info_context(ID, MSG, VERBOSITY, RO) \
  begin \
    if (RO.uvm_report_enabled(VERBOSITY, UVM_INFO, ID)) \
      RO.uvm_report_info(ID, MSG, VERBOSITY, `uvm_file, `uvm_line); \
  end

`define uvm_warning_context(ID, MSG, RO) \
  begin \
    if (RO.uvm_report_enabled(UVM_NONE, UVM_WARNING, ID)) \
      RO.uvm_report_warning(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

`define uvm_error_context(ID, MSG, RO) \
  begin \
    if (RO.uvm_report_enabled(UVM_NONE, UVM_ERROR, ID)) \
      RO.uvm_report_error(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

`define uvm_fatal_context(ID, MSG, RO) \
  begin \
    if (RO.uvm_report_enabled(UVM_NONE, UVM_FATAL, ID)) \
      RO.uvm_report_fatal(ID, MSG, UVM_NONE, `uvm_file, `uvm_line); \
  end

// Message begin/end macros
`define uvm_info_begin(ID, MSG, VERBOSITY, RM = __uvm_msg) \
  begin \
    uvm_report_message RM; \
    if (uvm_report_enabled(VERBOSITY, UVM_INFO, ID)) begin

`define uvm_info_end \
    end \
  end

`define uvm_warning_begin(ID, MSG, RM = __uvm_msg) \
  begin \
    uvm_report_message RM; \
    if (uvm_report_enabled(UVM_NONE, UVM_WARNING, ID)) begin

`define uvm_warning_end \
    end \
  end

`define uvm_error_begin(ID, MSG, RM = __uvm_msg) \
  begin \
    uvm_report_message RM; \
    if (uvm_report_enabled(UVM_NONE, UVM_ERROR, ID)) begin

`define uvm_error_end \
    end \
  end

`define uvm_fatal_begin(ID, MSG, RM = __uvm_msg) \
  begin \
    uvm_report_message RM; \
    if (uvm_report_enabled(UVM_NONE, UVM_FATAL, ID)) begin

`define uvm_fatal_end \
    end \
  end

// Message element macros
`define uvm_message_add_tag(NAME, VALUE, ACTION=UVM_LOG)

`define uvm_message_add_int(VAR, RADIX, LABEL="", ACTION=UVM_LOG)

`define uvm_message_add_string(VAR, LABEL="", ACTION=UVM_LOG)

`define uvm_message_add_object(VAR, LABEL="", ACTION=UVM_LOG)

// Report begin/end macros
`define uvm_report_begin(SEVERITY, ID, VERBOSITY, RO=uvm_get_report_object()) \
  begin

`define uvm_report_end \
  end

//===----------------------------------------------------------------------===//
// Callback Macros
//===----------------------------------------------------------------------===//

`define uvm_register_cb(T, CB)

`define uvm_do_callbacks(T, CB, METHOD)

`define uvm_do_callbacks_exit_on(T, CB, METHOD, VAL)

`define uvm_do_obj_callbacks(T, CB, OBJ, METHOD)

`define uvm_do_obj_callbacks_exit_on(T, CB, OBJ, METHOD, VAL)

//===----------------------------------------------------------------------===//
// Resource Macros
//===----------------------------------------------------------------------===//

`define uvm_resource_db_set(RSRCTYPE, SCOPE, NAME, VAL, ACCESSOR=null)

`define uvm_resource_db_get(RSRCTYPE, SCOPE, NAME, VAL, ACCESSOR=null)

`define uvm_resource_int(SCOPE, NAME, VAL)

`define uvm_resource_string(SCOPE, NAME, VAL)

//===----------------------------------------------------------------------===//
// Phase Macros
//===----------------------------------------------------------------------===//

`define uvm_phase_func_decl(PHASE) \
  virtual function void PHASE##_phase(uvm_phase phase); \
  endfunction

`define uvm_phase_task_decl(PHASE) \
  virtual task PHASE##_phase(uvm_phase phase); \
  endtask

`endif // UVM_MACROS_SVH
