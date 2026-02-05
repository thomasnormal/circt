// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test vtable dispatch through a deferred initialization pattern
// similar to UVM's uvm_deferred_init queue.

class wrapper;
  virtual function string get_type_name();
    return "<unknown>";
  endfunction
  virtual function void initialize();
    $display("wrapper::initialize called");
  endfunction
endclass

class registry #(type T=int, string Tname="<unknown>") extends wrapper;
  typedef registry #(T, Tname) this_type;
  static this_type m_inst;

  static function this_type get();
    if (m_inst == null)
      m_inst = new;
    return m_inst;
  endfunction

  virtual function string get_type_name();
    return Tname;
  endfunction

  virtual function void initialize();
    // CHECK-DAG: Registered: my_comp
    $display("Registered: %s", get_type_name());
  endfunction
endclass

// Deferred init queue (like uvm_deferred_init)
wrapper deferred_init[$];

class my_comp;
  typedef registry #(my_comp, "my_comp") type_id;
  // Static initializer pushes to deferred queue
  static bit m_init = do_deferred_init();
  static function bit do_deferred_init();
    deferred_init.push_back(type_id::get());
    return 1;
  endfunction
endclass

class my_other;
  typedef registry #(my_other, "my_other") type_id;
  static bit m_init = do_deferred_init();
  static function bit do_deferred_init();
    deferred_init.push_back(type_id::get());
    return 1;
  endfunction
endclass

module top;
  initial begin
    // Drain the deferred init queue (like uvm_init does)
    for (int i = 0; i < deferred_init.size(); i++) begin
      deferred_init[i].initialize();
    end

    // CHECK-DAG: Registered: my_other
    // CHECK: count = 2
    $display("count = %0d", deferred_init.size());
    // CHECK: PASS
    $display("PASS");
  end
endmodule
