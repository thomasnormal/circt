// RUN: not circt-sim %s --max-process-steps=0 --timeout=1 2>&1 | FileCheck %s

// CHECK: Wall-clock timeout reached

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %sig = llhd.sig %false : i1

  llhd.process {
  ^bb0:
    llhd.drv %sig, %false after %eps : i1
    cf.br ^bb1
  ^bb1:
    cf.br ^bb1
  }

  hw.output
}
