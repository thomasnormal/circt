// RUN: circt-verilog %s --ir-moore

class MyClass;
  int value;
endclass

interface test_if;
  MyClass obj;  // Class instance inside interface
endinterface

// Class that accesses interface member through virtual interface
class TestClass;
  virtual test_if vif;

  function void write_obj();
    MyClass c = new();
    c.value = 42;
    vif.obj = c;  // Write to class member through virtual interface
  endfunction

  function MyClass read_obj();
    return vif.obj;  // Read class member through virtual interface
  endfunction
endclass

module test;
  logic clk;
  virtual test_if vif;
  MyClass c;
  TestClass t;

  initial begin
    c = vif.obj;  // Direct access from module context
    vif.obj = c;  // Direct write from module context
    t = new();
    t.vif = vif;
    t.write_obj();  // Call method to prevent DCE
    c = t.read_obj();  // Call method to prevent DCE
  end
endmodule
