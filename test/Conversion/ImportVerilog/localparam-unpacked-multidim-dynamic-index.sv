// RUN: circt-verilog --ir-moore %s | FileCheck %s

module localparam_unpacked_multidim_dynamic_index(
    input logic [2:0] x,
    input logic [2:0] y,
    output logic [31:0] z);
  localparam int PiRotate [5][5] = '{
    '{0, 3, 1, 4, 2},
    '{1, 4, 2, 0, 3},
    '{2, 0, 3, 1, 4},
    '{3, 1, 4, 2, 0},
    '{4, 2, 0, 3, 1}
  };

  always_comb begin
    z = PiRotate[x][y];
  end
endmodule

// CHECK: moore.module @localparam_unpacked_multidim_dynamic_index
// CHECK: %[[TABLE:.+]] = moore.array_create %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : !moore.uarray<5 x i32>, !moore.uarray<5 x i32>, !moore.uarray<5 x i32>, !moore.uarray<5 x i32>, !moore.uarray<5 x i32> -> uarray<5 x uarray<5 x i32>>
// CHECK: %[[X:.+]] = moore.read %{{.+}} : <l3>
// CHECK: %[[XOFF:.+]] = moore.sub %{{.+}}, %[[X]] : l3
// CHECK: %[[ROW:.+]] = moore.dyn_extract %[[TABLE]] from %[[XOFF]] : uarray<5 x uarray<5 x i32>>, l3 -> uarray<5 x i32>
// CHECK: %[[Y:.+]] = moore.read %{{.+}} : <l3>
// CHECK: %[[YOFF:.+]] = moore.sub %{{.+}}, %[[Y]] : l3
// CHECK: %[[ELEM:.+]] = moore.dyn_extract %[[ROW]] from %[[YOFF]] : uarray<5 x i32>, l3 -> i32
// CHECK: %[[LOGIC:.+]] = moore.int_to_logic %[[ELEM]] : i32
// CHECK: moore.blocking_assign %z, %[[LOGIC]] : l32
