// RUN: circt-verilog %s --ir-moore | FileCheck %s
// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top test --max-time=1000000 2>&1 | FileCheck %s --check-prefix=SIM

// Test assignment of interface instance to virtual interface variable

interface simple_if;
  logic data;
  logic [7:0] value;
endinterface

// CHECK-LABEL: moore.module @test
module test;
  // CHECK: %intf = moore.interface.instance @simple_if : <virtual_interface<@simple_if>>
  simple_if intf();

  // CHECK: %vif = moore.variable : <virtual_interface<@simple_if
  virtual simple_if vif;

  initial begin
    // CHECK: [[VIF_LOCAL:%.*]] = moore.variable : <virtual_interface<@simple_if>>
    // CHECK: [[CONV:%.*]] = moore.conversion %intf : !moore.ref<virtual_interface<@simple_if>> -> !moore.virtual_interface
    // CHECK: moore.blocking_assign [[VIF_LOCAL]], [[CONV]]
    // CHECK: moore.blocking_assign %vif, {{.+}}
    vif = intf;

    // Access through virtual interface
    // CHECK: [[VIF_READ1:%.*]] = moore.read [[VIF_LOCAL]]
    // CHECK: moore.virtual_interface.signal_ref [[VIF_READ1]][@data]
    vif.data = 1;

    // CHECK: [[VIF_READ2:%.*]] = moore.read [[VIF_LOCAL]]
    // CHECK: moore.virtual_interface.signal_ref [[VIF_READ2]][@value]
    vif.value = 8'hAB;

    if (intf.data !== 1'b1 || intf.value !== 8'hAB) begin
      $display("SIM FAIL: virtual->interface write mismatch");
      $fatal;
    end

    intf.data = 1'b0;
    intf.value = 8'h3C;
    if (vif.data !== 1'b0 || vif.value !== 8'h3C) begin
      $display("SIM FAIL: interface->virtual read mismatch");
      $fatal;
    end

    $display("SIM PASS: virtual interface assignment");
  end
endmodule

// SIM: SIM PASS: virtual interface assignment
// SIM-NOT: SIM FAIL
// SIM-NOT: UVM_ERROR
// SIM-NOT: UVM_FATAL
