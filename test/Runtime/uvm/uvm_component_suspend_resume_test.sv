// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top tb_top --max-time=1000000000 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: suspend_freezes_ticks=PASS
// SIM: resume_advances_ticks=PASS
// SIM: UVM_COMPONENT_SUSPEND_RESUME_TEST_PASS

`timescale 1ns/1ps

`include "uvm_macros.svh"

package uvm_component_suspend_resume_test_pkg;
  import uvm_pkg::*;

  class ticking_component extends uvm_component;
    `uvm_component_utils(ticking_component)

    int ticks;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      ticks = 0;
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        #1;
        ticks++;
      end
    endtask
  endclass

  class suspend_resume_test extends uvm_test;
    `uvm_component_utils(suspend_resume_test)

    ticking_component c;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      c = ticking_component::type_id::create("c", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      int before_suspend;
      int during_suspend;
      int after_resume;

      phase.raise_objection(this, "suspend/resume regression");

      #20;
      before_suspend = c.ticks;

      c.suspend();
      #20;
      during_suspend = c.ticks;

      c.resume();
      #20;
      after_resume = c.ticks;

      // A single in-flight tick at suspend time is acceptable; ensure there
      // is no continued progress while suspended.
      if (during_suspend <= before_suspend + 1) begin
        $display("suspend_freezes_ticks=PASS");
      end else begin
        `uvm_error("COMP/SUSPEND",
                   $sformatf("ticks advanced while suspended: before=%0d during=%0d",
                             before_suspend, during_suspend))
      end

      if (after_resume > during_suspend) begin
        $display("resume_advances_ticks=PASS");
      end else begin
        `uvm_error("COMP/RESUME",
                   $sformatf("ticks did not advance after resume: during=%0d after=%0d",
                             during_suspend, after_resume))
      end

      if (during_suspend <= before_suspend + 1 && after_resume > during_suspend)
        $display("UVM_COMPONENT_SUSPEND_RESUME_TEST_PASS");

      phase.drop_objection(this, "suspend/resume regression");
    endtask
  endclass

endpackage

module tb_top;
  import uvm_pkg::*;
  import uvm_component_suspend_resume_test_pkg::*;

  initial begin
    run_test("suspend_resume_test");
  end
endmodule
