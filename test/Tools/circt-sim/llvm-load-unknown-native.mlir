// RUN: circt-sim %s | FileCheck %s

// CHECK: load=x

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "load="
  %fmt_nl = sim.fmt.literal "\0A"
  %c65536_i64 = llvm.mlir.constant(65536 : i64) : i64

  llhd.process {
    %ptr = llvm.inttoptr %c65536_i64 : i64 to !llvm.ptr
    %val = llvm.load %ptr : !llvm.ptr -> i1
    %fmt_val = sim.fmt.bin %val : i1
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
