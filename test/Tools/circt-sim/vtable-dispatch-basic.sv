// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test basic vtable dispatch: virtual method call on an object created
// via a static singleton pattern (similar to UVM factory registry).

class base_class;
  virtual function string get_name();
    return "base";
  endfunction
  virtual function int get_id();
    return 0;
  endfunction
endclass

class derived_class extends base_class;
  virtual function string get_name();
    return "derived";
  endfunction
  virtual function int get_id();
    return 42;
  endfunction
endclass

// Singleton pattern (like UVM registry)
class singleton;
  static derived_class m_inst;
  static function derived_class get();
    if (m_inst == null)
      m_inst = new;
    return m_inst;
  endfunction
endclass

module top;
  initial begin
    base_class obj;
    obj = singleton::get();

    // CHECK: name = derived
    $display("name = %s", obj.get_name());
    // CHECK: id = 42
    $display("id = %0d", obj.get_id());
    // CHECK: PASS
    $display("PASS");
  end
endmodule
