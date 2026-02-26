// RUN: not circt-lec --emit-mlir -c1=modA -c2=modB %s 2>&1 | FileCheck %s

module {
}

// CHECK: error: circuit 'modA' selected by -c1 was not found
// CHECK: error: no hw.module symbols were found in the input. This usually means frontend parsing failed upstream.
