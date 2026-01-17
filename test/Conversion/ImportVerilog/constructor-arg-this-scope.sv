// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Test that constructor argument evaluation uses the caller's 'this' scope,
// not the new object's type. This is a regression test for a bug where
// m_cb = new(..., m_cntxt) would incorrectly try to access m_cntxt on the
// new object's type instead of the caller's class.

class uvm_component;
  string name;
  function new(string n);
    name = n;
  endfunction
endclass

class uvm_heartbeat_callback;
  string cb_name;
  uvm_component cntxt;
  function new(string name, uvm_component c);
    cb_name = name;
    cntxt = c;
  endfunction
endclass

class uvm_heartbeat;
  protected uvm_heartbeat_callback m_cb;
  protected uvm_component m_cntxt;

  function new(string name, uvm_component cntxt);
    m_cntxt = cntxt;
    // BUG: When evaluating constructor arguments, 'this' should refer to
    // uvm_heartbeat (the caller), not uvm_heartbeat_callback (the new object).
    // m_cntxt is a property of uvm_heartbeat, not uvm_heartbeat_callback.
    m_cb = new({name,"_cb"}, m_cntxt);
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @uvm_heartbeat
// CHECK: func.func private @"uvm_heartbeat::new"
module top;
  uvm_component comp;
  uvm_heartbeat hb;
  initial begin
    comp = new("comp");
    hb = new("heartbeat", comp);
  end
endmodule
