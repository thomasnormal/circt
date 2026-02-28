// RUN: circt-verilog --parse-only --compat vcs %s
// Keep explicit string cast for compatibility across slang revisions.

module string_concat_byte;
  string s;
  byte b;

  initial begin
    s = "ab";
    b = s[0];
    s = {s, string'(b)};
  end
endmodule
