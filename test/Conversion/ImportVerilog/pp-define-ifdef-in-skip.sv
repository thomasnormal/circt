// RUN: circt-verilog --ir-moore --no-uvm-auto-include --single-unit %s %s
// REQUIRES: slang
//
// Regression test: When a `ifndef guard causes a `define body containing
// `ifdef/`else/`endif to be skipped (second inclusion via --single-unit),
// the preprocessor must not treat the `ifdef inside the macro body as a
// real conditional directive.

`ifndef PP_DEFINE_IFDEF_IN_SKIP_GUARD
`define PP_DEFINE_IFDEF_IN_SKIP_GUARD

`define ASSERT_MACRO(__name)                    \
`ifdef VERBOSE_MODE                              \
  $display("verbose: %s", `"__name`");          \
`else                                            \
  $display("quiet: %s", `"__name`");            \
`endif

`define NESTED_IFDEF_MACRO(__x)                  \
`ifdef FEAT_A                                    \
  `ifdef FEAT_B                                  \
    $display("A+B: %s", `"__x`");              \
  `else                                          \
    $display("A: %s", `"__x`");                \
  `endif                                         \
`else                                            \
  $display("none: %s", `"__x`");               \
`endif

module pp_define_ifdef_in_skip;
  initial begin
    `ASSERT_MACRO(hello)
    `NESTED_IFDEF_MACRO(world)
  end
endmodule

`endif // PP_DEFINE_IFDEF_IN_SKIP_GUARD
