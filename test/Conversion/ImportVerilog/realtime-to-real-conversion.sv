// RUN: circt-verilog %s --ir-moore | FileCheck %s

module RealtimeToRealConversion;
  function void sink(real x);
  endfunction

  initial begin
    sink($realtime());
  end
endmodule

// CHECK-LABEL: moore.module @RealtimeToRealConversion()
// CHECK:         moore.builtin.time
// CHECK:         moore.time_to_logic
// CHECK:         moore.logic_to_int {{.*}} : l64
// CHECK:         moore.uint_to_real {{.*}} : i64 -> f64
// CHECK:         moore.constant_real
// CHECK:         moore.fdiv
// CHECK:         func.call @sink
