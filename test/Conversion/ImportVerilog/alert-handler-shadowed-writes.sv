// RUN: circt-verilog --ir-hw %s
// Smoke test for shadowed register write patterns using alert_handler-style packages.

package prim_alert_pkg;
  typedef struct packed { logic alert_p; logic alert_n; } alert_tx_t;
  typedef struct packed { logic ping_p; logic ping_n; logic ack_p; logic ack_n; } alert_rx_t;
  parameter alert_tx_t ALERT_TX_DEFAULT = '{alert_p: 1'b0, alert_n: 1'b1};
endpackage

package prim_esc_pkg;
  typedef struct packed { logic esc_p; logic esc_n; } esc_tx_t;
  typedef struct packed { logic resp_p; logic resp_n; } esc_rx_t;
  parameter esc_rx_t ESC_RX_DEFAULT = '{resp_p: 1'b0, resp_n: 1'b1};
endpackage

package prim_mubi_pkg;
  typedef logic [3:0] mubi4_t;
  parameter mubi4_t MuBi4True = 4'h9;
endpackage

package edn_pkg;
  typedef struct packed { logic edn_req; } edn_req_t;
  typedef struct packed { logic edn_ack; logic edn_fips; logic [31:0] edn_bus; } edn_rsp_t;
  parameter edn_rsp_t EDN_RSP_DEFAULT = '0;
endpackage

package alert_handler_reg_pkg;
  localparam int unsigned NAlerts = 2;
  localparam int unsigned N_ESC_SEV = 2;
  localparam int unsigned NLpg = 1;
  parameter logic [10:0] ALERT_HANDLER_PING_TIMER_EN_SHADOWED_OFFSET = 11'h14;
  parameter logic [10:0] ALERT_HANDLER_ALERT_EN_SHADOWED_0_OFFSET = 11'h11c;
endpackage

package alert_handler_pkg;
  localparam int unsigned NAlerts = alert_handler_reg_pkg::NAlerts;
  localparam int unsigned N_ESC_SEV = alert_handler_reg_pkg::N_ESC_SEV;
  localparam int unsigned NLpg = alert_handler_reg_pkg::NLpg;
  typedef struct packed {
    logic [NAlerts-1:0] alert_cause;
  } alert_crashdump_t;
endpackage

module alert_handler_stub(
  input  logic clk_i,
  input  logic rst_ni,
  input  edn_pkg::edn_req_t edn_o,
  output edn_pkg::edn_rsp_t edn_i
);
  assign edn_i = edn_pkg::EDN_RSP_DEFAULT;
endmodule

module alert_handler_tb;
  import prim_alert_pkg::*;
  import prim_esc_pkg::*;
  import prim_mubi_pkg::*;
  import edn_pkg::*;
  import alert_handler_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  prim_mubi_pkg::mubi4_t [NLpg-1:0] lpg_cg_en_i = '{default: MuBi4True};
  prim_alert_pkg::alert_tx_t [NAlerts-1:0] alert_tx_i = '{default: ALERT_TX_DEFAULT};
  prim_esc_pkg::esc_rx_t [N_ESC_SEV-1:0] esc_rx_i = '{default: ESC_RX_DEFAULT};

  edn_req_t edn_o;
  edn_rsp_t edn_i;

  alert_handler_stub dut (
    .clk_i,
    .rst_ni,
    .edn_o,
    .edn_i
  );

  // Use package-qualified offsets to mimic shadowed writes.
  logic [10:0] ping_en_addr = alert_handler_reg_pkg::ALERT_HANDLER_PING_TIMER_EN_SHADOWED_OFFSET;
  logic [10:0] alert_en_addr = alert_handler_reg_pkg::ALERT_HANDLER_ALERT_EN_SHADOWED_0_OFFSET;
endmodule
