// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

module {
  moore.module @test_sscanf_time_string() {
    %timeout = moore.variable : <string>
    moore.procedure initial {
      %timeout_int = moore.variable : <time>
      %override_spec = moore.variable : <string>
      %0 = moore.constant_string "100,foo" : i56
      %1 = moore.int_to_string %0 : i56
      moore.blocking_assign %timeout, %1 : string
      %2 = moore.read %timeout : <string>
      %3 = moore.builtin.sscanf %2, "%d,%s", %timeout_int, %override_spec : !moore.ref<time>, !moore.ref<string>
      moore.return
    }
    moore.output
  }
}

// CHECK: llvm.func @__moore_sscanf
// CHECK: llvm.func @__moore_packed_string_to_string
// CHECK: llvm.call @__moore_sscanf
// CHECK: llvm.call @__moore_packed_string_to_string
// CHECK-NOT: moore.builtin.sscanf
