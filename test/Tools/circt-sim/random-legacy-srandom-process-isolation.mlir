// RUN: split-file %s %t
// RUN: circt-sim %t/base.mlir --top test 2>&1 | FileCheck %s --check-prefix=BASE
// RUN: circt-sim %t/interf.mlir --top test 2>&1 | FileCheck %s --check-prefix=INTERF
// RUN: circt-sim %t/base.mlir --top test 2>&1 | grep '^V=' > %t/base.out
// RUN: circt-sim %t/interf.mlir --top test 2>&1 | grep '^V=' > %t/interf.out
// RUN: diff %t/base.out %t/interf.out
//
// Legacy @srandom stubs (without object pointer) must be isolated per process.
// An unrelated process calling @srandom must not perturb another process's
// pending seed for the next __moore_randomize_basic call.

// BASE: V={{-?[0-9]+}}
// INTERF: V={{-?[0-9]+}}

//--- base.mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  func.func private @srandom(%seed: i32) {
    return
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %c8 = arith.constant 8 : i64
      %ptr = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr
      %zero32 = arith.constant 0 : i32
      %f0 = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      %f1 = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      llvm.store %zero32, %f0 : i32, !llvm.ptr
      llvm.store %zero32, %f1 : i32, !llvm.ptr

      %seed111 = arith.constant 111 : i32
      func.call @srandom(%seed111) : (i32) -> ()
      llhd.wait delay %t1, ^do
    ^do:
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c8)
          : (!llvm.ptr, i64) -> i32
      %val = llvm.load %f1 : !llvm.ptr -> i32
      %lit = sim.fmt.literal "V="
      %d = sim.fmt.dec %val signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %d, %nl)
      sim.proc.print %fmt
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}

//--- interf.mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  func.func private @srandom(%seed: i32) {
    return
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %c8 = arith.constant 8 : i64
      %ptr = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr
      %zero32 = arith.constant 0 : i32
      %f0 = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      %f1 = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      llvm.store %zero32, %f0 : i32, !llvm.ptr
      llvm.store %zero32, %f1 : i32, !llvm.ptr

      %seed111 = arith.constant 111 : i32
      func.call @srandom(%seed111) : (i32) -> ()
      llhd.wait delay %t1, ^do
    ^do:
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c8)
          : (!llvm.ptr, i64) -> i32
      %val = llvm.load %f1 : !llvm.ptr -> i32
      %lit = sim.fmt.literal "V="
      %d = sim.fmt.dec %val signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %d, %nl)
      sim.proc.print %fmt
      sim.terminate success, quiet
      llhd.halt
    }

    // Unrelated process; should not overwrite pending srandom seed of the
    // randomizing process above.
    llhd.process {
      %seed222 = arith.constant 222 : i32
      func.call @srandom(%seed222) : (i32) -> ()
      llhd.halt
    }

    hw.output
  }
}
