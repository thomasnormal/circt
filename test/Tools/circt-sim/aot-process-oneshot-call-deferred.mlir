// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=STATS
//
// One-shot processes with call sites are conservatively kept on interpreted
// dispatch. Native one-shot process callbacks bypass func.call/call_indirect
// interpreter policies today and can violate UVM safety assumptions.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
//
// SIM: call=1
//
// STATS: Compiled callback invocations:   0
// STATS: call=1

llvm.func @id_llvm(%x: i1) -> i1 {
  llvm.return %x : i1
}

hw.module @test() {
  %c0_i1 = hw.constant false
  %c1_i1 = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %fmt_prefix = sim.fmt.literal "call="
  %fmt_nl = sim.fmt.literal "\0A"
  %sig = llhd.sig %c0_i1 : i1

  // One-shot process: no llhd.wait, but includes llvm.call.
  llhd.process {
    llhd.drv %sig, %c1_i1 after %eps : i1
    %v = llvm.call @id_llvm(%c1_i1) : (i1) -> i1
    %v_dec = sim.fmt.dec %v : i1
    %msg = sim.fmt.concat (%fmt_prefix, %v_dec, %fmt_nl)
    sim.proc.print %msg
    llhd.halt
  }

  // End simulation after 1ns.
  llhd.process {
    %d = llhd.int_to_time %c1_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
