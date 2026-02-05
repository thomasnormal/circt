// RUN: circt-verilog %s --ir-hw 2>/dev/null | circt-sim 2>&1 | FileCheck %s

// Test that vtable entries are correctly populated using cached method
// implementations from resolveClassStructBody, even when ClassDeclOps may
// be erased before ClassNewOpConversion runs.
// This is a regression test for the vtable cache fix that ensures method
// implementations are preserved across the conversion pipeline.

class base_cls;
  virtual function string get_type();
    return "base";
  endfunction

  virtual function int get_id();
    return 0;
  endfunction
endclass

// mid_cls overrides get_type, inherits get_id from base
class mid_cls extends base_cls;
  virtual function string get_type();
    return "mid";
  endfunction
endclass

// leaf_cls overrides get_id, inherits get_type from mid
class leaf_cls extends mid_cls;
  virtual function int get_id();
    return 42;
  endfunction
endclass

// empty_cls inherits everything, overrides nothing
class empty_cls extends leaf_cls;
endclass

module vtable_cached_impl_test;
  initial begin
    base_cls b;
    mid_cls m;
    leaf_cls l;
    empty_cls e;
    base_cls ref_b;

    b = new();
    m = new();
    l = new();
    e = new();

    // Direct calls
    $display("b.get_type() = %s", b.get_type());
    $display("m.get_type() = %s", m.get_type());
    $display("l.get_type() = %s", l.get_type());
    $display("e.get_type() = %s", e.get_type());
    $display("e.get_id() = %0d", e.get_id());

    // Polymorphic calls through base reference
    ref_b = e;
    $display("ref_b.get_type() = %s", ref_b.get_type());
    $display("ref_b.get_id() = %0d", ref_b.get_id());

    $display("PASS");
    $finish;
  end
endmodule

// CHECK: b.get_type() = base
// CHECK: m.get_type() = mid
// CHECK: l.get_type() = mid
// CHECK: e.get_type() = mid
// CHECK: e.get_id() = 42
// CHECK: ref_b.get_type() = mid
// CHECK: ref_b.get_id() = 42
// CHECK: PASS
