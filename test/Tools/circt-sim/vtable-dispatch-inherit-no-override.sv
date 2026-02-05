// RUN: circt-verilog %s --ir-hw 2>/dev/null | circt-sim 2>&1 | FileCheck %s

// Test virtual method dispatch where a derived class inherits virtual
// methods without overriding them. This exercises the vtable population
// path in ClassNewOpConversion for classes without moore.vtable ops.

class animal;
  virtual function void speak();
    $display("animal");
  endfunction

  virtual function int legs();
    return 0;
  endfunction
endclass

// dog overrides speak but inherits legs
class dog extends animal;
  virtual function void speak();
    $display("woof");
  endfunction
endclass

// puppy doesn't override anything - inherits both from dog
class puppy extends dog;
  int age;
endclass

module vtable_inherit_no_override_test;
  initial begin
    puppy p;
    animal a;
    p = new();
    a = p;
    a.speak();       // should call dog::speak (inherited by puppy)
    $display("legs = %0d", a.legs());  // should call animal::legs (inherited by puppy)
    $display("PASS");
    $finish;
  end
endmodule

// CHECK: woof
// CHECK: legs = 0
// CHECK: PASS
