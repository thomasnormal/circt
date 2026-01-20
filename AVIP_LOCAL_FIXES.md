# AVIP Local Fixes (CIRCT Bring-Up)

This file tracks local edits applied to third-party AVIP sources to unblock
circt-verilog compilation. These changes are not intended for upstream.

## I2S AVIP
- Added `uvm_macros.svh` include + `import uvm_pkg::*;` in:
  - `/home/thomas-ahle/mbit/i2s_avip/src/hvlTop/i2sEnv/virtualSequencer/I2sVirtualSequencer.sv`
  - `/home/thomas-ahle/mbit/i2s_avip/src/hvlTop/tb/virtualSequences/I2sVirtualBaseSeq.sv`
  - `/home/thomas-ahle/mbit/i2s_avip/src/hvlTop/tb/test/I2sBaseTest.sv`
- Updated `uvm_test_done` usage to `uvm_test_done_objection::get()` in
  `/home/thomas-ahle/mbit/i2s_avip/src/hvlTop/tb/test/I2sBaseTest.sv`.
- Added `timescale 1ns/1ps` to all `*.sv` under
  `/home/thomas-ahle/mbit/i2s_avip/src`.

## AXI4Lite AVIP
- Added `uvm_macros.svh` include + `import uvm_pkg::*;` in
  `/home/thomas-ahle/mbit/axi4Lite_avip/src/axi4LiteEnv/virtualSequencer/Axi4LiteVirtualSequencer.sv`.
- Added `timescale 1ns/1ps` to all `*.sv` under
  `/home/thomas-ahle/mbit/axi4Lite_avip/src`.

## I3C AVIP
- Added `timescale 1ns/1ps` to all `*.sv` under
  `/home/thomas-ahle/mbit/i3c_avip/src`.

## APB AVIP
- Added `timescale 1ns/1ps` to all `*.sv` under
  `/home/thomas-ahle/mbit/apb_avip/src`.

## UART AVIP
- Removed default `uvm_comparer` argument from `do_compare` overrides in:
  - `/home/thomas-ahle/mbit/uart_avip/src/hvlTop/uartTxAgent/UartTxTransaction.sv`
  - `/home/thomas-ahle/mbit/uart_avip/src/hvlTop/uartRxAgent/UartRxTransaction.sv`

## JTAG AVIP
- Added `timescale 1ns/1ps` to:
  - `/home/thomas-ahle/mbit/jtag_avip/src/globals/JtagGlobalPkg.sv`
  - `/home/thomas-ahle/mbit/jtag_avip/src/hvlTop/jtagTargetDeviceAgent/JtagTargetDevicePkg.sv`
  - `/home/thomas-ahle/mbit/jtag_avip/src/hvlTop/jtagControllerDeviceAgent/JtagControllerDevicePkg.sv`
- Fixed enum casts and opcode indexing in
  `/home/thomas-ahle/mbit/jtag_avip/src/hdlTop/jtagTargetDeviceAgentBfm/JtagTargetDeviceDriverBfm.sv`.
- Removed default `uvm_comparer` argument from `do_compare` overrides in:
  - `/home/thomas-ahle/mbit/jtag_avip/src/hvlTop/jtagTargetDeviceAgent/JtagTargetDeviceTransaction.sv`
  - `/home/thomas-ahle/mbit/jtag_avip/src/hvlTop/jtagControllerDeviceAgent/JtagControllerDeviceTransaction.sv`

## UVM Stub Adjustments
- Added `timescale 1ns/1ps` to `/home/thomas-ahle/uvm-core/src/uvm_pkg.sv`.
- Added `uvm_test_done_objection` + `uvm_test_done` in `lib/Runtime/uvm/uvm_pkg.sv`.
