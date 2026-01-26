// RUN: circt-verilog --timescale=1ns/1ns %s --ir-moore | FileCheck %s

// Test mixed static/dynamic streaming concatenation operators.
// See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".

module test;
  int i_header;
  int i_len;
  byte i_data[];
  int i_crc;

  int o_header;
  int o_len;
  byte o_data[];
  int o_crc;

  // Mixed streaming: static prefix + dynamic array + static suffix
  // CHECK: moore.stream_concat_mixed
  // CHECK-SAME: slice 8
  initial begin : test_mixed_stream_concat
    byte pkt[$];

    i_header = 12;
    i_len = 5;
    i_crc = 42;
    i_data = new[5];

    pkt = {<< 8 {i_header, i_len, i_data, i_crc}};
  end

  // Mixed streaming unpack: static prefix + dynamic array + static suffix
  // CHECK: moore.stream_unpack_mixed
  // CHECK-SAME: slice 8
  initial begin : test_mixed_stream_unpack
    byte pkt[$];

    {<< 8 {o_header, o_len, o_data, o_crc}} = pkt;
  end

  // Single dynamic array streaming (no mixed operands)
  // CHECK: moore.stream_concat
  initial begin : test_single_dynamic_stream
    byte pkt[$];
    byte data[];

    pkt = {<< 8 {data}};

    // Single dynamic array unpack
    // CHECK: moore.stream_unpack
    {<< 8 {data}} = pkt;
  end

endmodule
