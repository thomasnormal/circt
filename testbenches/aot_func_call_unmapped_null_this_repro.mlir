llvm.mlir.global internal @dummy(0 : i8) : i8

func.func private @"uvm_pkg::uvm_component::get_children"(%this: !llvm.ptr, %q: !llvm.ptr) {
  %v = llvm.load %this : !llvm.ptr -> i8
  llvm.store %v, %q : i8, !llvm.ptr
  return
}

hw.module @top() {
  %one = hw.constant 1 : i64
  %prefix = sim.fmt.literal "safe="
  %nl = sim.fmt.literal "\0A"
  %gp = llvm.mlir.addressof @dummy : !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %ok = hw.constant 1 : i32

  llhd.process {
    func.call @"uvm_pkg::uvm_component::get_children"(%null, %gp) : (!llvm.ptr, !llvm.ptr) -> ()
    %v = sim.fmt.dec %ok signed : i32
    %msg = sim.fmt.concat (%prefix, %v, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %one
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
