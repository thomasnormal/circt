// RUN: split-file %s %t
// RUN: circt-sim %t/base.mlir --top test 2>&1 | FileCheck %s --check-prefix=BASE
// RUN: circt-sim %t/interf.mlir --top test 2>&1 | FileCheck %s --check-prefix=INTERF
// RUN: circt-sim %t/base.mlir --top test 2>&1 | grep '^W=' > %t/base.out
// RUN: circt-sim %t/interf.mlir --top test 2>&1 | grep '^W=' > %t/interf.out
// RUN: diff %t/base.out %t/interf.out
//
// __moore_randomize_with_range* should use the last randomized object from the
// current process. Randomization in another process must not perturb this.
//
// BASE: W={{-?[0-9]+}}
// INTERF: W={{-?[0-9]+}}

//--- base.mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
  llvm.func @__moore_randomize_with_range(i64, i64) -> i64

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
      %seed111 = arith.constant 111 : i32
      func.call @srandom(%seed111) : (i32) -> ()
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c8)
          : (!llvm.ptr, i64) -> i32
      llhd.wait delay %t1, ^do
    ^do:
      %zero64 = arith.constant 0 : i64
      %hund64 = arith.constant 100 : i64
      %w = llvm.call @__moore_randomize_with_range(%zero64, %hund64)
          : (i64, i64) -> i64
      %lit = sim.fmt.literal "W="
      %d = sim.fmt.dec %w signed : i64
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
  llvm.func @__moore_randomize_with_range(i64, i64) -> i64

  func.func private @srandom(%seed: i32) {
    return
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %t05_i64 = arith.constant 500000 : i64
    %t05 = llhd.int_to_time %t05_i64
    llhd.process {
      cf.br ^start
    ^start:
      %c8 = arith.constant 8 : i64
      %ptr = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr
      %seed111 = arith.constant 111 : i32
      func.call @srandom(%seed111) : (i32) -> ()
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c8)
          : (!llvm.ptr, i64) -> i32
      llhd.wait delay %t1, ^do
    ^do:
      %zero64 = arith.constant 0 : i64
      %hund64 = arith.constant 100 : i64
      %w = llvm.call @__moore_randomize_with_range(%zero64, %hund64)
          : (i64, i64) -> i64
      %lit = sim.fmt.literal "W="
      %d = sim.fmt.dec %w signed : i64
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %d, %nl)
      sim.proc.print %fmt
      sim.terminate success, quiet
      llhd.halt
    }

    // Unrelated process randomization must not overwrite the last randomized
    // object tracked for the process above.
    llhd.process {
      cf.br ^s
    ^s:
      %c8b = arith.constant 8 : i64
      %ptrb = llvm.call @malloc(%c8b) : (i64) -> !llvm.ptr
      %seed222 = arith.constant 222 : i32
      func.call @srandom(%seed222) : (i32) -> ()
      llhd.wait delay %t05, ^do2
    ^do2:
      %rc2 = llvm.call @__moore_randomize_basic(%ptrb, %c8b)
          : (!llvm.ptr, i64) -> i32
      llhd.halt
    }

    hw.output
  }
}
