// RUN: circt-verilog --parse-only --compat vcs %s
// Class handle $display formatting - class handles are converted to 64-bit
// integers (zero placeholder) when used in format strings.

class fmt_handle;
endclass

module format_class_handle;
  fmt_handle h;
  initial begin
    h = null;
    $display("%0d %0h", h, null);
  end
endmodule
