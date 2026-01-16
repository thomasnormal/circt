module test_display;
  initial begin
    $display("Hello from CIRCT!");
    $display("Testing: %d + %d = %d", 2, 3, 5);
  end
endmodule
