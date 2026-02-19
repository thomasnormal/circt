// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd | circt-sim --top top --mode=compile --jit-compile-budget=100000 --jit-report=%t.jit.json | FileCheck %s
// RUN: FileCheck %s --check-prefix=JIT < %t.jit.json

typedef struct packed {
  byte writeData;
} payload_t;

class foo_monitor_bfm;
  task sampleWriteDataAndACK(inout payload_t p);
    fork
      begin
        p.writeData = 8'hfb;
        #1;
      end
    join_none
    wait fork;
  endtask
endclass

module top;
  payload_t p;
  foo_monitor_bfm m;

  initial begin
    m = new();
    p.writeData = 8'h00;
    m.sampleWriteDataAndACK(p);
    $display("writeData=%0h", p.writeData);
    $finish;
  end
endmodule

// CHECK: writeData=fb
// JIT: "detail": "monitor_sample_write_data_fork_child_interpreter_fallback"
