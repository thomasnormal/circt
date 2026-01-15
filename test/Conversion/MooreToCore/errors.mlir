// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics

// Note: queue<string> is now supported, so the previous test case for
// "invalid type" was removed.

func.func @unsupportedConversion() {
    %0 = moore.constant_string "Test" : i32
    // expected-error @below {{failed to legalize operation 'moore.conversion'}}
    %1 = moore.conversion %0 : !moore.i32 -> !moore.string
  return
}
