// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 --vcd %t.vcd > %t.out 2>&1
// RUN: /usr/bin/grep -q "Simulation completed" %t.out
// RUN: bash -eu -c '\
// RUN:   sig_ids=$(/usr/bin/grep "__sva__cover_immediate_" %t.vcd | /usr/bin/awk '\''{ print $4 }'\''); \
// RUN:   test -n "$sig_ids"; \
// RUN:   for sig_id in $sig_ids; do /usr/bin/grep -q "0$sig_id" %t.vcd; done; \
// RUN:   hit=0; \
// RUN:   for sig_id in $sig_ids; do \
// RUN:     if /usr/bin/grep -q "1$sig_id" %t.vcd; then hit=1; fi; \
// RUN:   done; \
// RUN:   test "$hit" -eq 1'
//
// Runtime immediate covers should surface as synthetic 1-bit VCD signals so
// waveform viewers can show cover-hit transitions for procedural cover sites.

module top;
  reg a;

  initial begin
    a = 1'b0;
    cover (a);      // miss
    a = 1'b1;
    cover (a);      // hit
    #1;
    $finish;
  end
endmodule
