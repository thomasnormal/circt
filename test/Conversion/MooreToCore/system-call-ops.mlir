// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test system call operations: $strobe, $monitor, $monitoron, $monitoroff,
// $printtimescale, $ferror, $fseek, $rewind, $fread, $readmemb, $readmemh

//===----------------------------------------------------------------------===//
// $strobe Operation - prints at end of time step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_strobe
moore.module @test_strobe(in %fmt: !moore.format_string) {
  // CHECK: sim.print_formatted_proc
  moore.builtin.strobe %fmt
  moore.output
}

//===----------------------------------------------------------------------===//
// $fstrobe Operation - writes to file at end of time step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fstrobe
moore.module @test_fstrobe(in %fd: !moore.i32, in %fmt: !moore.format_string) {
  // CHECK: sim.print_formatted_proc
  moore.builtin.fstrobe %fd, %fmt
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitor Operation - enables continuous monitoring
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitor
moore.module @test_monitor(in %fmt: !moore.format_string) {
  // CHECK: sim.print_formatted_proc
  moore.builtin.monitor %fmt
  moore.output
}

//===----------------------------------------------------------------------===//
// $fmonitor Operation - continuous monitoring to file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fmonitor
moore.module @test_fmonitor(in %fd: !moore.i32, in %fmt: !moore.format_string) {
  // CHECK: sim.print_formatted_proc
  moore.builtin.fmonitor %fd, %fmt
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitoron Operation - enables monitoring
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitoron
moore.module @test_monitoron() {
  // monitoron is a no-op in the lowering, it gets erased
  // CHECK-NOT: monitoron
  moore.builtin.monitoron
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitoroff Operation - disables monitoring
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitoroff
moore.module @test_monitoroff() {
  // monitoroff is a no-op in the lowering, it gets erased
  // CHECK-NOT: monitoroff
  moore.builtin.monitoroff
  moore.output
}

//===----------------------------------------------------------------------===//
// $printtimescale Operation - prints timescale info
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_printtimescale
moore.module @test_printtimescale() {
  // printtimescale is a no-op in the lowering, it gets erased
  // CHECK-NOT: printtimescale
  moore.builtin.printtimescale
  moore.output
}

//===----------------------------------------------------------------------===//
// $ferror Operation - returns file error status
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_ferror
moore.module @test_ferror(in %fd: !moore.i32, out err: !moore.i32) {
  %strVar = moore.variable : <string>
  // ferror returns 0 (stub)
  // CHECK: hw.constant 0 : i32
  %err = moore.builtin.ferror %fd, %strVar : <string>
  moore.output %err : !moore.i32
}

//===----------------------------------------------------------------------===//
// $fseek Operation - seek file position
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fseek
moore.module @test_fseek(in %fd: !moore.i32, in %offset: !moore.i32, in %op: !moore.i32, out result: !moore.i32) {
  // fseek returns 0 (success stub)
  // CHECK: hw.constant 0 : i32
  %r = moore.builtin.fseek %fd, %offset, %op
  moore.output %r : !moore.i32
}

//===----------------------------------------------------------------------===//
// $rewind Operation - rewind file position
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_rewind
moore.module @test_rewind(in %fd: !moore.i32) {
  // rewind is erased (no-op stub)
  // CHECK-NOT: rewind
  moore.builtin.rewind %fd
  moore.output
}

//===----------------------------------------------------------------------===//
// $fread Operation - read binary data from file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fread
moore.module @test_fread(in %fd: !moore.i32, out bytes: !moore.i32) {
  %var = moore.variable : <i32>
  // fread returns 0 (stub)
  // CHECK: hw.constant 0 : i32
  %b = moore.builtin.fread %var, %fd : <i32>
  moore.output %b : !moore.i32
}

//===----------------------------------------------------------------------===//
// $readmemb Operation - load memory from binary file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_readmemb
moore.module @test_readmemb(in %filename: !moore.string) {
  %mem = moore.variable : <array<8 x i8>>
  // readmemb is erased (no-op stub)
  // CHECK-NOT: readmemb
  moore.builtin.readmemb %filename, %mem : <array<8 x i8>>
  moore.output
}

//===----------------------------------------------------------------------===//
// $readmemh Operation - load memory from hex file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_readmemh
moore.module @test_readmemh(in %filename: !moore.string) {
  %mem = moore.variable : <array<8 x i8>>
  // readmemh is erased (no-op stub)
  // CHECK-NOT: readmemh
  moore.builtin.readmemh %filename, %mem : <array<8 x i8>>
  moore.output
}

//===----------------------------------------------------------------------===//
// $ungetc Operation - push a character back
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_ungetc
moore.module @test_ungetc(in %c: !moore.i32, in %fd: !moore.i32, out result: !moore.i32) {
  // ungetc returns the pushed character
  // CHECK: [[C:%.*]] = comb.concat %{{.*}}
  // CHECK: hw.output [[C]]
  %r = moore.builtin.ungetc %c, %fd
  moore.output %r : !moore.i32
}
