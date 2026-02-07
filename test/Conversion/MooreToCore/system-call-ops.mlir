// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test system call operations: $strobe, $monitor, $monitoron, $monitoroff,
// $printtimescale, $ferror, $fseek, $rewind, $fread, $readmemb, $readmemh, $ungetc

//===----------------------------------------------------------------------===//
// $strobe Operation - prints at end of time step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_strobe
moore.module @test_strobe() {
  // CHECK: seq.initial
  // CHECK:   sim.proc.print
  moore.procedure initial {
    %fmt = moore.fmt.literal "strobe test"
    moore.builtin.strobe %fmt
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $fstrobe Operation - writes to file at end of time step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fstrobe
moore.module @test_fstrobe(in %fd: !moore.i32) {
  // CHECK: seq.initial
  // CHECK:   sim.proc.print
  moore.procedure initial {
    %fmt = moore.fmt.literal "fstrobe test"
    %c1 = moore.constant 1 : i32
    moore.builtin.fstrobe %c1, %fmt
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitor Operation - enables continuous monitoring
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitor
moore.module @test_monitor() {
  // CHECK: seq.initial
  // CHECK:   sim.proc.print
  moore.procedure initial {
    %fmt = moore.fmt.literal "monitor test"
    moore.builtin.monitor %fmt
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $fmonitor Operation - continuous monitoring to file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fmonitor
moore.module @test_fmonitor() {
  // CHECK: seq.initial
  // CHECK:   sim.proc.print
  moore.procedure initial {
    %fmt = moore.fmt.literal "fmonitor test"
    %c1 = moore.constant 1 : i32
    moore.builtin.fmonitor %c1, %fmt
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitoron Operation - enables monitoring (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitoron
moore.module @test_monitoron() {
  // monitoron is erased; since procedure becomes empty, initial block is also removed
  // CHECK-NOT: monitoron
  // CHECK: hw.output
  moore.procedure initial {
    moore.builtin.monitoron
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $monitoroff Operation - disables monitoring (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_monitoroff
moore.module @test_monitoroff() {
  // monitoroff is erased; since procedure becomes empty, initial block is also removed
  // CHECK-NOT: monitoroff
  // CHECK: hw.output
  moore.procedure initial {
    moore.builtin.monitoroff
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $printtimescale Operation - prints timescale info (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_printtimescale
moore.module @test_printtimescale() {
  // printtimescale is erased; since procedure becomes empty, initial block is also removed
  // CHECK-NOT: printtimescale
  // CHECK: hw.output
  moore.procedure initial {
    moore.builtin.printtimescale
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $ferror Operation - returns file error status
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_ferror
moore.module @test_ferror(out err: !moore.i32) {
  // CHECK: hw.constant 0 : i32
  %result = moore.variable : <i32>
  moore.procedure initial {
    %fd = moore.constant 1 : i32
    %strVar = moore.variable : <string>
    %err = moore.builtin.ferror %fd, %strVar : <string>
    moore.blocking_assign %result, %err : i32
    moore.return
  }
  %read = moore.read %result : <i32>
  moore.output %read : !moore.i32
}

//===----------------------------------------------------------------------===//
// $fseek Operation - seek file position
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fseek
moore.module @test_fseek(out result: !moore.i32) {
  // CHECK: hw.constant 0 : i32
  %result = moore.variable : <i32>
  moore.procedure initial {
    %fd = moore.constant 1 : i32
    %offset = moore.constant 0 : i32
    %op = moore.constant 0 : i32
    %r = moore.builtin.fseek %fd, %offset, %op
    moore.blocking_assign %result, %r : i32
    moore.return
  }
  %read = moore.read %result : <i32>
  moore.output %read : !moore.i32
}

//===----------------------------------------------------------------------===//
// $rewind Operation - rewind file position (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_rewind
moore.module @test_rewind() {
  // rewind is erased; since procedure becomes empty, initial block is also removed
  // CHECK-NOT: rewind
  // CHECK: hw.output
  moore.procedure initial {
    %fd = moore.constant 1 : i32
    moore.builtin.rewind %fd
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// $fread Operation - read binary data from file
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fread
moore.module @test_fread(out bytes: !moore.i32) {
  // CHECK: hw.constant 0 : i32
  %result = moore.variable : <i32>
  moore.procedure initial {
    %fd = moore.constant 1 : i32
    %var = moore.variable : <i32>
    %b = moore.builtin.fread %var, %fd : <i32>
    moore.blocking_assign %result, %b : i32
    moore.return
  }
  %read = moore.read %result : <i32>
  moore.output %read : !moore.i32
}

//===----------------------------------------------------------------------===//
// $readmemb Operation - load memory from binary file (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_readmemb
// CHECK: llvm.call @__moore_readmemb
// CHECK-NOT: moore.builtin.readmemb
func.func @test_readmemb(%filename: !moore.string) {
  %mem = moore.variable : <uarray<8 x i8>>
  moore.builtin.readmemb %filename, %mem : <uarray<8 x i8>>
  return
}

//===----------------------------------------------------------------------===//
// $readmemh Operation - load memory from hex file (no-op, erased)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_readmemh
// CHECK: llvm.call @__moore_readmemh
// CHECK-NOT: moore.builtin.readmemh
func.func @test_readmemh(%filename: !moore.string) {
  %mem = moore.variable : <uarray<8 x i8>>
  moore.builtin.readmemh %filename, %mem : <uarray<8 x i8>>
  return
}

//===----------------------------------------------------------------------===//
// $ungetc Operation - push a character back
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_ungetc
moore.module @test_ungetc(out result: !moore.i32) {
  // ungetc returns the input character
  // CHECK: hw.constant 65 : i32
  %result = moore.variable : <i32>
  moore.procedure initial {
    %c = moore.constant 65 : i32
    %fd = moore.constant 1 : i32
    %r = moore.builtin.ungetc %c, %fd
    moore.blocking_assign %result, %r : i32
    moore.return
  }
  %read = moore.read %result : <i32>
  moore.output %read : !moore.i32
}
