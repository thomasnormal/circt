// RUN: circt-opt --hw-externalize-modules='module-names=foo,bar' %s | FileCheck %s --check-prefix=EXTERNALIZE
// RUN: not circt-opt --hw-externalize-modules='module-names=missing' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: circt-opt --hw-externalize-modules='module-names=missing allow-missing=true' %s | FileCheck %s --check-prefix=ALLOW

module {
  hw.module @foo(in %in: i1, out out: i1) {
    hw.output %in : i1
  }

  hw.module @bar() {
    hw.output
  }

  hw.module @baz(in %in: i1, out out: i1) {
    hw.output %in : i1
  }
}

// EXTERNALIZE: hw.module.extern @foo(in %in : i1, out out : i1)
// EXTERNALIZE: hw.module.extern @bar()
// EXTERNALIZE: hw.module @baz
// EXTERNALIZE-NOT: hw.module @foo
// EXTERNALIZE-NOT: hw.module @bar

// MISSING: error: missing requested modules: missing

// ALLOW: hw.module @foo
// ALLOW: hw.module @bar
// ALLOW: hw.module @baz
