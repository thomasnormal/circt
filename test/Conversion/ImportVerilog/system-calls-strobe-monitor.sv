// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test SystemVerilog system calls: $strobe, $fstrobe, $monitor, $fmonitor,
// $monitoron, $monitoroff, $printtimescale, $fflush, $ftell, $ungetc

// CHECK-LABEL: moore.module @StrobeMonitorTest
module StrobeMonitorTest;
  int fd, x, y, pos, ch;

  initial begin
    // Test $strobe - display at end of time step
    // CHECK: moore.builtin.strobe
    $strobe("strobe: x=%d", x);

    // Test $strobeb - strobe with binary format
    // CHECK: moore.builtin.strobe
    $strobeb(x);

    // Test $strobeo - strobe with octal format
    // CHECK: moore.builtin.strobe
    $strobeo(x);

    // Test $strobeh - strobe with hex format
    // CHECK: moore.builtin.strobe
    $strobeh(x);

    // Test $fstrobe - file strobe
    // CHECK: moore.builtin.fstrobe
    $fstrobe(fd, "fstrobe: y=%d", y);

    // Test $fstrobeb - file strobe with binary format
    // CHECK: moore.builtin.fstrobe
    $fstrobeb(fd, y);

    // Test $monitor - continuous monitoring
    // CHECK: moore.builtin.monitor
    $monitor("monitor: x=%d y=%d", x, y);

    // Test $monitorb - monitor with binary format
    // CHECK: moore.builtin.monitor
    $monitorb(x);

    // Test $fmonitor - file monitoring
    // CHECK: moore.builtin.fmonitor
    $fmonitor(fd, "fmonitor: x=%d", x);

    // Test $monitoron - enable monitoring
    // CHECK: moore.builtin.monitoron
    $monitoron;

    // Test $monitoroff - disable monitoring
    // CHECK: moore.builtin.monitoroff
    $monitoroff;

    // Test $printtimescale - print timescale
    // CHECK: moore.builtin.printtimescale
    $printtimescale;

    // Test $fflush - flush file buffer
    // CHECK: moore.builtin.fflush
    $fflush(fd);

    // Test $fflush without args - flush all files
    // CHECK: moore.constant 0 : i32
    // CHECK: moore.builtin.fflush
    $fflush();

    // Test $ftell - get file position
    // CHECK: moore.builtin.ftell
    pos = $ftell(fd);

    // Test $ungetc - push character back to file
    // CHECK: moore.builtin.ungetc
    ch = $ungetc(ch, fd);
  end
endmodule
