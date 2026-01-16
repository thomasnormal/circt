// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// End-to-end Class Lowering Test
//===----------------------------------------------------------------------===//
// This test verifies that class-based SystemVerilog code parses correctly
// to Moore dialect. It tests:
// - Forward-declared class types (the P0 fix)
// - Class properties and methods
// - Class inheritance and upcasting
// - Queue of class handles
// - Class instantiation and method calls
// - Null comparison for class handles

//===----------------------------------------------------------------------===//
// Forward-declared class types (P0 fix verification)
//===----------------------------------------------------------------------===//

// Queue of forward-declared class handles - appears first in output
// MOORE: moore.global_variable @"pkg::items" : !moore.queue<class<@"pkg::Item">, 0>

// The class is forward-declared via typedef, then used in a queue
// MOORE-LABEL: moore.class.classdecl @"pkg::Item" {
// MOORE:   moore.class.propertydecl @id : !moore.i32
// MOORE:   moore.class.propertydecl @data : !moore.l8
// MOORE: }

// MOORE-LABEL: func.func private @"pkg::Item::new"
// MOORE: moore.class.property_ref %arg0[@id]
// MOORE: moore.class.property_ref %arg0[@data]

// MOORE-LABEL: func.func private @"pkg::Item::get_id"
// MOORE: moore.class.property_ref %arg0[@id]
// MOORE: moore.read

package pkg;
  // Forward declare the class before use
  typedef class Item;

  // Queue of forward-declared class handles
  Item items[$];

  // The actual class definition
  class Item;
    int id;
    logic [7:0] data;

    function new(int id_val);
      id = id_val;
      data = 8'h00;
    endfunction

    function int get_id();
      return id;
    endfunction

    function void set_data(logic [7:0] d);
      data = d;
    endfunction
  endclass
endpackage

//===----------------------------------------------------------------------===//
// Class with properties and methods
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.class.classdecl @Counter {
// MOORE:   moore.class.propertydecl @count : !moore.i32
// MOORE:   moore.class.propertydecl @max_count : !moore.i32
// MOORE: }
class Counter;
  int count;
  int max_count;

  function new(int max_val);
    count = 0;
    max_count = max_val;
  endfunction

  function void increment();
    if (count < max_count)
      count = count + 1;
  endfunction

  function void reset();
    count = 0;
  endfunction

  function int get_count();
    return count;
  endfunction
endclass

// MOORE-LABEL: func.func private @"Counter::new"
// MOORE-SAME: (%arg0: !moore.class<@Counter>, %arg1: !moore.i32)
// MOORE: moore.class.property_ref %arg0[@count]
// MOORE: moore.class.property_ref %arg0[@max_count]

// MOORE-LABEL: func.func private @"Counter::increment"
// MOORE-SAME: (%arg0: !moore.class<@Counter>)
// MOORE: moore.class.property_ref %arg0[@count]
// MOORE: moore.class.property_ref %arg0[@max_count]
// MOORE: moore.slt
// MOORE: moore.add

// MOORE-LABEL: func.func private @"Counter::get_count"
// MOORE-SAME: (%arg0: !moore.class<@Counter>) -> !moore.i32
// MOORE: moore.class.property_ref %arg0[@count]
// MOORE: moore.read

//===----------------------------------------------------------------------===//
// Class inheritance hierarchy
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.class.classdecl @BaseClass {
// MOORE:   moore.class.propertydecl @base_val : !moore.i32
// MOORE: }
class BaseClass;
  int base_val;

  function new();
    base_val = 0;
  endfunction

  function int get_base();
    return base_val;
  endfunction
endclass

// MOORE-LABEL: moore.class.classdecl @DerivedClass extends @BaseClass {
// MOORE:   moore.class.propertydecl @derived_val : !moore.i32
// MOORE: }
class DerivedClass extends BaseClass;
  int derived_val;

  function new();
    super.new();
    derived_val = 100;
  endfunction

  function int get_derived();
    return derived_val;
  endfunction

  function int get_sum();
    return base_val + derived_val;
  endfunction
endclass

// MOORE-LABEL: func.func private @"DerivedClass::new"
// MOORE-SAME: (%arg0: !moore.class<@DerivedClass>)
// MOORE: moore.class.upcast %arg0 : <@DerivedClass> to <@BaseClass>
// MOORE: call @"BaseClass::new"

// MOORE-LABEL: func.func private @"DerivedClass::get_sum"
// MOORE-SAME: (%arg0: !moore.class<@DerivedClass>) -> !moore.i32
// MOORE: moore.class.upcast %arg0 : <@DerivedClass> to <@BaseClass>
// MOORE: moore.class.property_ref {{.*}}[@base_val]
// MOORE: moore.class.property_ref %arg0[@derived_val]
// MOORE: moore.add

//===----------------------------------------------------------------------===//
// Queue of class handles (tests forward-declared class in queue)
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.class.classdecl @Container {
// MOORE:   moore.class.propertydecl @counters : !moore.queue<class<@Counter>, 0>
// MOORE: }
class Container;
  Counter counters[$];

  function void add_counter(int max_val);
    Counter c;
    c = new(max_val);
    counters.push_back(c);
  endfunction

  function int count_items();
    return counters.size();
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Module using classes - verifies instantiation and method calls
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.module @ClassE2ETest()
module ClassE2ETest;
  // Class handle variables
  // MOORE: %ctr = moore.variable : <class<@Counter>>
  Counter ctr;

  // MOORE: %derived = moore.variable : <class<@DerivedClass>>
  DerivedClass derived;

  // MOORE: %base = moore.variable : <class<@BaseClass>>
  BaseClass base;

  // Result variables
  // MOORE: %val = moore.variable : <i32>
  int val;

  // MOORE: moore.procedure initial {
  initial begin
    // Test class instantiation with constructor
    // MOORE: moore.class.new : <@Counter>
    // MOORE: func.call @"Counter::new"
    ctr = new(10);

    // Test method call
    // MOORE: func.call @"Counter::increment"
    ctr.increment();
    ctr.increment();
    ctr.increment();

    // Test property read via method
    // MOORE: func.call @"Counter::get_count"
    val = ctr.get_count();

    // Test inheritance - create derived, assign to base (upcast)
    // MOORE: moore.class.new : <@DerivedClass>
    derived = new();

    // Polymorphic assignment (implicit upcast)
    // MOORE: moore.class.upcast {{.*}} : <@DerivedClass> to <@BaseClass>
    base = derived;

    // Test method on derived class
    // MOORE: func.call @"DerivedClass::get_sum"
    val = derived.get_sum();

    // Test inherited property access via method
    // MOORE: moore.class.upcast
    // MOORE: func.call @"BaseClass::get_base"
    val = derived.get_base();
  end
  // MOORE: moore.output
endmodule

//===----------------------------------------------------------------------===//
// Module testing forward-declared class queue
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.module @ForwardDeclTest()
module ForwardDeclTest;
  import pkg::*;

  // MOORE: %item = moore.variable : <class<@"pkg::Item">>
  Item item;
  int id_val;

  // MOORE: moore.procedure initial {
  initial begin
    // Create item using forward-declared class
    // MOORE: moore.class.new : <@"pkg::Item">
    // MOORE: func.call @"pkg::Item::new"
    item = new(42);

    // Get property via method
    // MOORE: func.call @"pkg::Item::get_id"
    id_val = item.get_id();

    // Add to global queue of forward-declared class handles
    // MOORE: moore.get_global_variable @"pkg::items"
    // MOORE: moore.queue.push_back
    items.push_back(item);
  end
endmodule

//===----------------------------------------------------------------------===//
// Module testing null comparison
//===----------------------------------------------------------------------===//

// MOORE-LABEL: moore.module @NullCompareTest()
module NullCompareTest;
  // MOORE: %c = moore.variable : <class<@Counter>>
  Counter c;
  int result;

  // MOORE: moore.procedure initial {
  initial begin
    // Test null comparison before assignment
    // MOORE: moore.read %c
    // MOORE: moore.class.null : <@Counter>
    // MOORE: moore.class_handle_cmp eq
    if (c == null) begin
      result = 0;
    end

    // Create object
    // MOORE: moore.class.new : <@Counter>
    c = new(5);

    // Test not-null comparison
    // MOORE: moore.class_handle_cmp ne
    if (c != null) begin
      result = 1;
    end
  end
endmodule
