// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module BoundedDelay(input logic clk, req, ack, data, valid);
  // CHECK-LABEL: moore.module @BoundedDelay

  // ##[1:3] - bounded delay range (1 to 3 cycles)
  property bounded_req_ack;
    @(posedge clk) req |-> ##[1:3] ack;
  endproperty
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 1, 2 : i1
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (bounded_req_ack);

  // ##[0:2] - bounded delay starting at 0
  property zero_bounded;
    @(posedge clk) req |-> ##[0:2] ack;
  endproperty
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 2 : i1
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (zero_bounded);

  // ##[*] - unbounded delay (0 or more cycles)
  property unbounded_delay;
    @(posedge clk) req |-> ##[*] ack;
  endproperty
  // CHECK: [[DELAYSTAR:%[a-z0-9]+]] = ltl.delay {{%[a-z0-9]+}}, 0 : i1
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[DELAYSTAR]] : i1, !ltl.sequence
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (unbounded_delay);

  // ##[+] - positive delay (1 or more cycles)
  property positive_delay;
    @(posedge clk) req |-> ##[+] ack;
  endproperty
  // CHECK: [[DELAYPLUS:%[a-z0-9]+]] = ltl.delay {{%[a-z0-9]+}}, 1 : i1
  // CHECK: ltl.implication {{%[a-z0-9]+}}, [[DELAYPLUS]] : i1, !ltl.sequence
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (positive_delay);

  // ##[2:5] larger range
  property larger_range;
    @(posedge clk) data |-> ##[2:5] valid;
  endproperty
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 2, 3 : i1
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (larger_range);

  // Chained bounded delays
  property chained_bounded;
    @(posedge clk) req ##[1:2] data ##[1:3] ack;
  endproperty
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 0 : i1
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 1 : i1
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 2 : i1
  // CHECK: ltl.concat
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (chained_bounded);
endmodule
