// RUN: circt-verilog --timescale=1ns/1ns %s --ir-moore | FileCheck %s

// Test mixed static/dynamic streaming with large static prefixes (> 64 bits).
// This tests the fix for the 64-bit static prefix limit issue.
// See IEEE 1800-2017 Section 11.4.14 "Streaming operators (pack/unpack)".

// CHECK-LABEL: moore.module @test
module test;
  // Test with 96-bit static prefix (header + len + crc = 32+32+32 = 96 bits)
  int i_header;
  int i_len;
  int i_crc;
  byte i_data[];

  int o_header;
  int o_len;
  int o_crc;
  byte o_data[];

  initial begin : test_96bit_prefix
    byte pkt[$];

    i_header = 12;
    i_len = 5;
    i_crc = 42;
    i_data = new[5];

    // 96-bit static prefix (exceeds 64-bit limit)
    // CHECK: moore.stream_concat_mixed[{{.*}}, {{.*}}, {{.*}}] {{.*}}[]
    // CHECK-SAME: slice 8 : (!moore.i32, !moore.i32, !moore.i32, !moore.open_uarray<i8>)
    pkt = {<< 8 {i_header, i_len, i_crc, i_data}};
  end

  initial begin : test_96bit_prefix_unpack
    byte pkt[$];

    // 96-bit static prefix unpack
    // CHECK: moore.stream_unpack_mixed[{{.*}}, {{.*}}, {{.*}}] {{.*}}[]
    // CHECK-SAME: slice 8 : (!moore.ref<i32>, !moore.ref<i32>, !moore.ref<i32>
    {<< 8 {o_header, o_len, o_crc, o_data}} = pkt;
  end

  // Test with 128-bit static prefix (4 ints = 4*32 = 128 bits)
  int a, b, c, d;
  byte arr[];

  initial begin : test_128bit_prefix
    byte pkt[$];

    // 128-bit static prefix
    // CHECK: moore.stream_concat_mixed[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {{.*}}[]
    // CHECK-SAME: slice 8 : (!moore.i32, !moore.i32, !moore.i32, !moore.i32, !moore.open_uarray<i8>)
    pkt = {<< 8 {a, b, c, d, arr}};
  end

  // Test with 64-bit static prefix (exactly at the old limit)
  int x, y;
  byte data[];

  initial begin : test_64bit_prefix
    byte pkt[$];

    // 64-bit static prefix (at the old limit boundary)
    // CHECK: moore.stream_concat_mixed[{{.*}}, {{.*}}] {{.*}}[]
    // CHECK-SAME: slice 8 : (!moore.i32, !moore.i32, !moore.open_uarray<i8>)
    pkt = {<< 8 {x, y, data}};
  end

  // Test with suffix only (no prefix)
  int suffix1, suffix2, suffix3;
  byte middle_data[];

  initial begin : test_96bit_suffix
    byte pkt[$];

    // 96-bit static suffix
    // CHECK: moore.stream_concat_mixed[] {{.*}}[{{.*}}, {{.*}}, {{.*}}]
    // CHECK-SAME: slice 8 : (!moore.open_uarray<i8>, !moore.i32, !moore.i32, !moore.i32)
    pkt = {<< 8 {middle_data, suffix1, suffix2, suffix3}};
  end

  // Test with both large prefix and suffix
  int p1, p2, p3;
  int s1, s2, s3;
  byte both_data[];

  initial begin : test_large_prefix_and_suffix
    byte pkt[$];

    // Both 96-bit prefix and 96-bit suffix
    // CHECK: moore.stream_concat_mixed[{{.*}}, {{.*}}, {{.*}}] {{.*}}[{{.*}}, {{.*}}, {{.*}}]
    // CHECK-SAME: slice 8 : (!moore.i32, !moore.i32, !moore.i32, !moore.open_uarray<i8>, !moore.i32, !moore.i32, !moore.i32)
    pkt = {<< 8 {p1, p2, p3, both_data, s1, s2, s3}};
  end

endmodule
