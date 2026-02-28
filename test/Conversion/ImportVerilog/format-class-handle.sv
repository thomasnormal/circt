// RUN: circt-verilog --parse-only --compat vcs %s
// Class handle $display formatting using pointer-compatible specifiers.

class fmt_handle;
endclass

module format_class_handle;
  fmt_handle h;
  initial begin
    h = null;
    $display("%0p %0p", h, null);
  end
endmodule
