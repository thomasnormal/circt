// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $coverage_control, $coverage_get_max, $coverage_merge, $coverage_save.
// Bug: All coverage system tasks return 0 unconditionally.
// IEEE 1800-2017 Section 20.13:
// - $coverage_control should return 1 on success
// - $coverage_get_max should return non-zero max coverage value
module top;
  integer ret;

  initial begin
    // $coverage_control(control_type, coverage_type, scope_def, modules_or_instance)
    // control_type=1 (start), coverage_type=1 (toggle), scope_def=0 (local)
    ret = $coverage_control(1, 1, 0, top);
    // Should return 1 (success), not 0
    // CHECK: coverage_ctrl=1
    $display("coverage_ctrl=%0d", ret);

    // $coverage_get_max should return a non-zero value representing max coverage
    ret = $coverage_get_max(1, 0, top);
    // CHECK-NOT: coverage_max=0
    $display("coverage_max=%0d", ret);

    $finish;
  end
endmodule
