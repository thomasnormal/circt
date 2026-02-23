module {
  llvm.mlir.global internal constant @__plusarg_MISSING("MISSING\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @__plusarg_DEBUG("DEBUG\00") {addr_space = 0 : i32}
  llvm.func @__moore_test_plusargs(!llvm.ptr, i32) -> i32
  llvm.mlir.global internal constant @__plusarg_VERBOSE("VERBOSE\00") {addr_space = 0 : i32}
  hw.module @top() {
    %0 = sim.fmt.literal "missing_not_found\0A"
    %1 = sim.fmt.literal "missing_found\0A"
    %2 = sim.fmt.literal "debug_not_found\0A"
    %3 = sim.fmt.literal "debug_found\0A"
    %4 = sim.fmt.literal "verbose_not_found\0A"
    %5 = sim.fmt.literal "verbose_found\0A"
    %c0_i32 = hw.constant 0 : i32
    %6 = llvm.mlir.addressof @__plusarg_MISSING : !llvm.ptr
    %7 = llvm.mlir.constant(5 : i32) : i32
    %8 = llvm.mlir.addressof @__plusarg_DEBUG : !llvm.ptr
    %9 = llvm.mlir.constant(7 : i32) : i32
    %10 = llvm.mlir.addressof @__plusarg_VERBOSE : !llvm.ptr
    llhd.process {
      %11 = llvm.call @__moore_test_plusargs(%10, %9) : (!llvm.ptr, i32) -> i32
      %12 = comb.icmp ne %11, %c0_i32 : i32
      cf.cond_br %12, ^bb1, ^bb2
    ^bb1:
      sim.proc.print %5
      cf.br ^bb3
    ^bb2:
      sim.proc.print %4
      cf.br ^bb3
    ^bb3:
      %13 = llvm.call @__moore_test_plusargs(%8, %7) : (!llvm.ptr, i32) -> i32
      %14 = comb.icmp ne %13, %c0_i32 : i32
      cf.cond_br %14, ^bb4, ^bb5
    ^bb4:
      sim.proc.print %3
      cf.br ^bb6
    ^bb5:
      sim.proc.print %2
      cf.br ^bb6
    ^bb6:
      %15 = llvm.call @__moore_test_plusargs(%6, %9) : (!llvm.ptr, i32) -> i32
      %16 = comb.icmp ne %15, %c0_i32 : i32
      cf.cond_br %16, ^bb7, ^bb8
    ^bb7:
      sim.proc.print %1
      cf.br ^bb9
    ^bb8:
      sim.proc.print %0
      cf.br ^bb9
    ^bb9:
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
