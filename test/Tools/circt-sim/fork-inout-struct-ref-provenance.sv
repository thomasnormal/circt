// RUN: circt-verilog %s --no-uvm-auto-include --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top fork_inout_struct_ref_provenance 2>&1 | FileCheck %s
//
// Regression: inout struct refs passed through a function call must remain
// resolvable in join_none children even after the parent frame returns.
// Without this, delayed child drives fail with unresolved llhd.drv.

`timescale 1ns/1ps

typedef struct {
  int cnt;
  bit [7:0] data;
} pkt_t;

class conv;
  virtual task automatic bump(inout pkt_t p);
    fork
      begin
        #1;
        p.cnt = p.cnt + 1;
        p.data = 8'hA5;
      end
    join_none
  endtask
endclass

module fork_inout_struct_ref_provenance;
  pkt_t pkt;
  conv c;

  initial begin
    pkt = '{default:0};
    c = new();
    c.bump(pkt);
    #2;

    // CHECK: cnt=1 data=a5
    $display("cnt=%0d data=%0h", pkt.cnt, pkt.data);

    if (pkt.cnt != 1 || pkt.data != 8'hA5) begin
      $display("FAIL");
      $fatal(1);
    end

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
