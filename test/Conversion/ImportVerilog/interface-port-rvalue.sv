// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that interface port signals can be used as rvalues inside interface tasks.
// This is a common pattern in UVM BFM (Bus Functional Model) interfaces where
// tasks need to wait on clock edges and check signal values.

// CHECK-LABEL: func.func private @"test_bfm::wait_for_reset"
// CHECK-SAME: (%arg0: !moore.virtual_interface<@test_bfm>)
// CHECK: moore.virtual_interface.signal_ref %arg0[@preset_n]
// CHECK: moore.detect_event negedge
interface test_bfm (
    input bit pclk,
    input bit preset_n,
    input bit pready,
    output logic penable
);

    // Task that uses interface port signals as rvalues (edge events)
    task wait_for_reset();
        @(negedge preset_n);  // preset_n used as rvalue (edge detection)
        @(posedge preset_n);
    endtask

    // CHECK-LABEL: func.func private @"test_bfm::check_ready"
    // CHECK-SAME: (%arg0: !moore.virtual_interface<@test_bfm>)
    // CHECK: moore.virtual_interface.signal_ref %arg0[@pready]
    // CHECK: moore.read
    task check_ready();
        while (pready == 0) begin  // pready used as rvalue
            @(posedge pclk);
        end
    endtask

    // CHECK-LABEL: func.func private @"test_bfm::drive_output"
    // CHECK-SAME: (%arg0: !moore.virtual_interface<@test_bfm>)
    // CHECK: moore.virtual_interface.signal_ref %arg0[@penable]
    // CHECK: moore.nonblocking_assign
    task drive_output();
        penable <= 1'b1;  // output port as lvalue
    endtask

endinterface : test_bfm

// CHECK-LABEL: moore.module @top
// CHECK: %bfm_inst = moore.interface.instance @test_bfm
// CHECK: moore.procedure initial
// CHECK: moore.read %bfm_inst
// CHECK: func.call @"test_bfm::wait_for_reset"

// Top module to instantiate the interface and call tasks
module top;
    bit clk;
    bit rst_n;
    bit ready;

    test_bfm bfm_inst (
        .pclk(clk),
        .preset_n(rst_n),
        .pready(ready),
        .penable()
    );

    // Call the interface tasks to prevent them from being eliminated
    initial begin
        bfm_inst.wait_for_reset();
        bfm_inst.check_ready();
        bfm_inst.drive_output();
    end
endmodule
