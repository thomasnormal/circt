// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  typedef struct {
    int count;
    byte data;
  } pkt_t;

  task automatic fill_pkt(output pkt_t pkt);
    // Intentionally leave `count` unassigned to verify output-arg defaults.
    pkt.data = 8'h2a;
  endtask

  initial begin
    pkt_t pkt;
    pkt.count = 41;
    pkt.data = 0;
    fill_pkt(pkt);
    // CHECK: count=0 data=42
    $display("count=%0d data=%0d", pkt.count, pkt.data);
    $finish;
  end
endmodule
