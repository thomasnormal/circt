// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test TLM FIFO with two producers, one consumer.

// CHECK: [TEST] consumer got all 10 items: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_fifo_item extends uvm_object;
    `uvm_object_utils(edge_fifo_item)
    int producer_id;
    int seq_num;
    function new(string name = "edge_fifo_item");
      super.new(name);
    endfunction
  endclass

  class edge_fifo_intlv_test extends uvm_test;
    `uvm_component_utils(edge_fifo_intlv_test)
    uvm_tlm_fifo #(edge_fifo_item) fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("fifo", this, 16);
    endfunction

    task run_phase(uvm_phase phase);
      int count;
      phase.raise_objection(this);

      // Fork two producers
      fork
        begin // Producer 0
          for (int i = 0; i < 5; i++) begin
            edge_fifo_item item = new($sformatf("p0_%0d", i));
            item.producer_id = 0;
            item.seq_num = i;
            fifo.put(item);
            #1;
          end
        end
        begin // Producer 1
          for (int i = 0; i < 5; i++) begin
            edge_fifo_item item = new($sformatf("p1_%0d", i));
            item.producer_id = 1;
            item.seq_num = i;
            fifo.put(item);
            #1;
          end
        end
      join

      // Consumer gets all 10
      count = 0;
      while (fifo.used() > 0) begin
        edge_fifo_item got;
        fifo.get(got);
        count++;
      end

      if (count == 10)
        `uvm_info("TEST", "consumer got all 10 items: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("consumer got %0d items: FAIL", count))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_fifo_intlv_test");
endmodule
