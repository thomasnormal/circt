// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-string-put-value-delayed-test.c -ldl
// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top vpi_string_delay_test --max-time=100000 --vpi=%t.vpi.so 2>&1 | FileCheck %s
//
// CHECK: VPI_STRING_DELAY: asciival_sum=1668
// CHECK: VPI_STRING_DELAY: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_STRING_DELAY: FINAL: {{[0-9]+}} passed, 0 failed

module vpi_string_delay_test(
    input string stream_in_string,
    output int stream_in_string_asciival_sum
);
  string stream_in_string_asciival_str;
  int stream_in_string_asciival;

  always @(stream_in_string) begin
    stream_in_string_asciival_sum = 0;
    for (int idx = 0; idx < stream_in_string.len(); idx++) begin
      stream_in_string_asciival_str = $sformatf("%0d", stream_in_string[idx]);
      stream_in_string_asciival = stream_in_string_asciival_str.atoi();
      stream_in_string_asciival_sum += stream_in_string_asciival;
    end
  end
endmodule
