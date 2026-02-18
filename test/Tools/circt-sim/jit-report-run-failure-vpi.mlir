// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: not circt-sim %s --skip-passes --mode=compile --vpi=%t/missing-vpi-library.so --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Failed to load VPI library: {{.*}}missing-vpi-library.so
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 0
// JSON: "jit_deopts_total": 0

hw.module @top() {
  hw.output
}
