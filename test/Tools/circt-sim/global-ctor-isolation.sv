// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that global constructors are isolated from each other.
// When one constructor causes the process to halt (e.g., due to a virtual
// method call on an uninitialized vtable), subsequent constructors must
// still execute correctly. This pattern is critical for UVM factory
// registration where uvm_component_utils expands to static initializers.

class registry_base;
  virtual function string get_type_name();
    return "";
  endfunction
endclass

// This class's static init will run first and call a virtual method
// that may fail with X due to vtable ordering
class early_registry extends registry_base;
  static early_registry m_inst;

  // Singleton getter called from static initializer
  static function early_registry get();
    if (m_inst == null) begin
      m_inst = new;
    end
    return m_inst;
  endfunction

  virtual function string get_type_name();
    return "early";
  endfunction
endclass

// Queue to track registration order (like uvm_deferred_init)
string registered_names[$];

// This class's static init runs later and must not be blocked
class late_registry extends registry_base;
  static late_registry m_inst;

  static function late_registry get();
    if (m_inst == null) begin
      m_inst = new;
    end
    return m_inst;
  endfunction

  virtual function string get_type_name();
    return "late";
  endfunction
endclass

// Static initializers (like uvm_component_utils expansion)
// These run as global constructors
bit early_initialized = register_type(early_registry::get());
bit late_initialized = register_type(late_registry::get());

function automatic bit register_type(registry_base obj);
  registered_names.push_back(obj.get_type_name());
  return 1;
endfunction

module top;
  initial begin
    // CHECK: registered count = 2
    $display("registered count = %0d", registered_names.size());
    // CHECK: PASS
    if (registered_names.size() == 2)
      $display("PASS");
    else
      $display("FAIL: expected 2 registrations");
  end
endmodule
