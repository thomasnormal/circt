// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test unique constraint with narrow range to force collision resolution.
// With x, y, z in [1:5] and unique constraint, collisions are very likely
// without enforcement.
module top;
  class item;
    rand int x;
    rand int y;
    rand int z;
    constraint c_range { x >= 1; x <= 5; y >= 1; y <= 5; z >= 1; z <= 5; }
    constraint c_unique { unique {x, y, z}; }

    function new();
      x = 0; y = 0; z = 0;
    endfunction
  endclass

  initial begin
    item obj = new();
    int all_unique;
    int i;

    all_unique = 1;
    process::self().srandom(42);
    for (i = 0; i < 30; i++) begin
      void'(obj.randomize());
      if (obj.x == obj.y || obj.x == obj.z || obj.y == obj.z)
        all_unique = 0;
    end

    // CHECK: all_unique=1
    $display("all_unique=%0d", all_unique);

    $finish;
  end
endmodule
