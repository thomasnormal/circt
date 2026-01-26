// RUN: circt-verilog --parse-only --compat vcs %s
// VCS compatibility mode now supports mixing strings and integers in concatenation.

module string_concat_byte;
  string s;
  byte b;

  initial begin
    s = "ab";
    b = s[0];
    s = {s, b};
  end
endmodule
