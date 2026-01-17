// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test for Issue: Recursive class conversion during function body conversion
// should not use stale currentThisRef from enclosing function.
//
// This is a regression test for a bug where converting class A's method body
// would set currentThisRef to class A, then trigger conversion of class B
// (via type references), and class B's constraint expressions would incorrectly
// try to upcast class A's 'this' to class B, causing:
// "receiver class is not the same as, or derived from, expected base class"
//
// The fix is to save and clear currentThisRef when entering ClassDeclVisitor::run,
// so that nested class conversions don't inherit a stale 'this' reference.

// CHECK-NOT: error:
// CHECK-NOT: is not the same as, or derived from

// Class with constraints (like UVM's uvm_mem_mam_policy)
class constrained_class;
  rand int unsigned offset;
  int unsigned min_val;
  int unsigned max_val;

  // Constraint that accesses instance properties
  constraint valid_range {
    offset >= min_val;
    offset <= max_val;
  }
endclass

// Class with out-of-line method that references constrained_class
class outer_class;
  extern function void process_constrained(constrained_class obj);
endclass

// Out-of-line implementation - this sets currentThisRef to outer_class
// and may trigger conversion of constrained_class
function void outer_class::process_constrained(constrained_class obj);
  if (obj != null) begin
    $display("Processing constrained object");
  end
endfunction

// CHECK: moore.class.classdecl @constrained_class
// CHECK: moore.constraint.block @valid_range
// CHECK: moore.class.classdecl @outer_class
