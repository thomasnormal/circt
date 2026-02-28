// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-llhd --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top uvm_string_to_severity_test --max-time=1000000000 2>&1 | FileCheck %s --check-prefix=SIM

// SIM-NOT: FAIL:
// SIM: PASS: uvm_string_to_severity

`include "uvm_macros.svh"
import uvm_pkg::*;

module uvm_string_to_severity_test;
  initial begin
    uvm_severity sev;
    bit ok;

    ok = uvm_string_to_severity("UVM_INFO", sev);
    if (!ok || sev != UVM_INFO) begin
      $display("FAIL: UVM_INFO parse mismatch");
      $finish;
    end

    ok = uvm_string_to_severity("UVM_WARNING", sev);
    if (!ok || sev != UVM_WARNING) begin
      $display("FAIL: UVM_WARNING parse mismatch");
      $finish;
    end

    ok = uvm_string_to_severity("UVM_ERROR", sev);
    if (!ok || sev != UVM_ERROR) begin
      $display("FAIL: UVM_ERROR parse mismatch");
      $finish;
    end

    ok = uvm_string_to_severity("UVM_FATAL", sev);
    if (!ok || sev != UVM_FATAL) begin
      $display("FAIL: UVM_FATAL parse mismatch");
      $finish;
    end

    ok = uvm_string_to_severity("NOT_A_SEVERITY", sev);
    if (ok) begin
      $display("FAIL: invalid severity should not parse");
      $finish;
    end

    $display("PASS: uvm_string_to_severity");
    $finish;
  end
endmodule
