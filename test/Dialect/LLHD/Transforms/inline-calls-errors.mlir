// RUN: circt-opt --llhd-inline-calls --verify-diagnostics --split-input-file %s

hw.module @CallInGraphRegion() {
  // expected-error @below {{function call cannot be inlined in this region}}
  func.call @foo() : () -> ()
}

func.func @foo() {
  cf.br ^bb1
^bb1:
  return
}

// -----

hw.module @RecursiveCalls() {
  llhd.combinational {
    func.call @foo() : () -> ()
    llhd.yield
  }
}

// expected-note @below {{callee is foo}}
func.func @foo() {
  call @bar() : () -> ()
  return
}

func.func @bar() {
  // expected-error @below {{recursive function call cannot be inlined (unsupported in --ir-hw)}}
  call @foo() : () -> ()
  return
}
