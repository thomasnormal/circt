// RUN: circt-verilog --help | FileCheck %s

// CHECK: OVERVIEW: Verilog and SystemVerilog frontend
// CHECK-DAG: --language-version
// CHECK-DAG: --max-parse-depth
// CHECK-DAG: --max-lexer-errors
// CHECK-DAG: --num-threads
// CHECK-DAG: --max-instance-depth
// CHECK-DAG: --max-generate-steps
// CHECK-DAG: --max-constexpr-depth
// CHECK-DAG: --max-constexpr-steps
// CHECK-DAG: --max-constexpr-backtrace
// CHECK-DAG: --max-instance-array
// CHECK-DAG: --disable-local-includes
// CHECK-DAG: --enable-legacy-protect
// CHECK-DAG: --map-keyword-version
// CHECK-DAG: --translate-off-format
// CHECK-DAG: --timing
// CHECK-DAG: --allow-hierarchical-const
// CHECK-DAG: --relax-enum-conversions
// CHECK-DAG: --relax-string-conversions
// CHECK-DAG: --allow-recursive-implicit-call
// CHECK-DAG: --allow-bare-value-param-assignment
// CHECK-DAG: --allow-self-determined-stream-concat
// CHECK-DAG: --allow-merging-ansi-ports
// CHECK-DAG: --allow-top-level-iface-ports
