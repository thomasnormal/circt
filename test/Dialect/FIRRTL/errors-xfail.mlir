// RUN: circt-opt %s | FileCheck %s

// Previously this would fail with "could not determine domain-type of destination".
// The issue was fixed. See: https://github.com/llvm/circt/issues/9398

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  firrtl.domain @ClockDomain
  firrtl.domain @PowerDomain
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo(
    in %in: !firrtl.domain of @ClockDomain,
    out %out: !firrtl.domain of @PowerDomain
  ) {
    // CHECK: %w = firrtl.wire : !firrtl.domain
    %w = firrtl.wire : !firrtl.domain
    // CHECK: firrtl.domain.define %w, %in
    firrtl.domain.define %w, %in
    // CHECK: firrtl.domain.define %out, %w
    firrtl.domain.define %out, %w
  }
}
