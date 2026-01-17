// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test covergroup transition bins conversion to Moore dialect ops.
// IEEE 1800-2017 Section 19.5.4 "Defining transition bins"

module test_covergroup_transition_bins;
  logic [2:0] state;
  logic       clk;

  // FSM states
  localparam IDLE = 3'd0;
  localparam RUN  = 3'd1;
  localparam DONE = 3'd2;
  localparam ERR  = 3'd3;

  // Test simple transition bins
  // CHECK: moore.covergroup.decl @cg_transitions sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @idle_to_run kind<bins> transitions {{\[\[\[}}0, 0, 0, 0], [1, 0, 0, 0]]]
  // CHECK:     moore.coverbin.decl @run_to_done kind<bins> transitions {{\[\[\[}}1, 0, 0, 0], [2, 0, 0, 0]]]
  // CHECK:   }
  // CHECK: }
  covergroup cg_transitions @(posedge clk);
    state_cp: coverpoint state {
      bins idle_to_run = (IDLE => RUN);
      bins run_to_done = (RUN => DONE);
    }
  endgroup

  // Test multi-step transition sequence
  // CHECK: moore.covergroup.decl @cg_sequence sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @full_cycle kind<bins> transitions {{\[\[\[}}0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]]
  // CHECK:   }
  // CHECK: }
  covergroup cg_sequence @(posedge clk);
    state_cp: coverpoint state {
      bins full_cycle = (IDLE => RUN => DONE => IDLE);
    }
  endgroup

  // Test multiple alternative transitions in one bin
  // CHECK: moore.covergroup.decl @cg_alternatives sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @to_error kind<bins> transitions {{\[\[\[}}0, 0, 0, 0], [3, 0, 0, 0]], {{\[\[}}1, 0, 0, 0], [3, 0, 0, 0]], {{\[\[}}2, 0, 0, 0], [3, 0, 0, 0]]]
  // CHECK:   }
  // CHECK: }
  covergroup cg_alternatives @(posedge clk);
    state_cp: coverpoint state {
      // Multiple alternative sequences in one bin
      bins to_error = (IDLE => ERR), (RUN => ERR), (DONE => ERR);
    }
  endgroup

  // Test transition with repeat patterns (consecutive)
  // CHECK: moore.covergroup.decl @cg_repeat sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @triple_idle kind<bins> transitions {{\[\[\[}}0, 1, 3, 3], [1, 0, 0, 0]]]
  // CHECK:   }
  // CHECK: }
  covergroup cg_repeat @(posedge clk);
    state_cp: coverpoint state {
      // Consecutive repeat: stay in IDLE for exactly 3 cycles, then go to RUN
      bins triple_idle = (IDLE [*3] => RUN);
    }
  endgroup

  // Test illegal transition bins
  // CHECK: moore.covergroup.decl @cg_illegal_trans sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @bad_trans kind<illegal_bins> transitions {{\[\[\[}}3, 0, 0, 0], [0, 0, 0, 0]]]
  // CHECK:   }
  // CHECK: }
  covergroup cg_illegal_trans @(posedge clk);
    state_cp: coverpoint state {
      illegal_bins bad_trans = (ERR => IDLE);  // Going from error to idle is illegal
    }
  endgroup

  // Test default sequence bin
  // CHECK: moore.covergroup.decl @cg_default_seq sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l3 {
  // CHECK:     moore.coverbin.decl @known_trans kind<bins> transitions {{\[\[\[}}0, 0, 0, 0], [1, 0, 0, 0]], {{\[\[}}1, 0, 0, 0], [2, 0, 0, 0]]]
  // CHECK:     moore.coverbin.decl @other_trans kind<bins> default default_sequence
  // CHECK:   }
  // CHECK: }
  covergroup cg_default_seq @(posedge clk);
    state_cp: coverpoint state {
      bins known_trans = (IDLE => RUN), (RUN => DONE);
      bins other_trans = default sequence;  // Catch all other transitions
    }
  endgroup

endmodule
