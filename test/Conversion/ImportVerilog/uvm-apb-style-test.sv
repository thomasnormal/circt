// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
// XFAIL: *

// Test UVM stubs with APB AVIP testbench style patterns

`include "uvm_macros.svh"

// Global parameters package (like APB AVIP's apb_global_pkg)
package global_pkg;
  parameter int NO_OF_SLAVES = 1;
  parameter int MASTER_AGENT_ACTIVE = 1;
  parameter int SLAVE_AGENT_ACTIVE = 1;
  parameter int ADDRESS_WIDTH = 32;
  parameter int DATA_WIDTH = 32;

  typedef enum bit[31:0] {
    BIT_8  = 32'd8,
    BIT_16 = 32'd16,
    BIT_24 = 32'd24,
    BIT_32 = 32'd32
  } transfer_size_e;

  typedef enum bit {
    NO_ERROR = 1'b0,
    ERROR    = 1'b1
  } slave_error_e;

  typedef enum bit {
    WRITE = 1'b1,
    READ  = 1'b0
  } tx_type_e;

  typedef enum logic [2:0] {
    NORMAL_SECURE_DATA           = 3'b000,
    NORMAL_SECURE_INSTRUCTION    = 3'b001,
    NORMAL_NONSECURE_DATA        = 3'b010,
    NORMAL_NONSECURE_INSTRUCTION = 3'b011
  } protection_type_e;

endpackage : global_pkg

// Master agent config (like APB AVIP's apb_master_agent_config)
package master_pkg;
  `include "uvm_macros.svh"
  import uvm_pkg::*;
  import global_pkg::*;

  // Agent configuration class
  class master_agent_config extends uvm_object;
    `uvm_object_utils(master_agent_config)

    uvm_active_passive_enum is_active = UVM_ACTIVE;
    int no_of_slaves = 1;
    bit has_coverage = 1;
    bit [ADDRESS_WIDTH-1:0] master_min_addr_range_array[16];
    bit [ADDRESS_WIDTH-1:0] master_max_addr_range_array[16];
    bit [ADDRESS_WIDTH-1:0] paddr;

    function new(string name = "master_agent_config");
      super.new(name);
    endfunction

    function void master_min_addr_range(int index, bit [ADDRESS_WIDTH-1:0] addr);
      master_min_addr_range_array[index] = addr;
    endfunction

    function void master_max_addr_range(int index, bit [ADDRESS_WIDTH-1:0] addr);
      master_max_addr_range_array[index] = addr;
    endfunction
  endclass

  // Transaction class (like APB AVIP's apb_master_tx)
  class master_tx extends uvm_sequence_item;
    `uvm_object_utils(master_tx)

    rand bit [ADDRESS_WIDTH-1:0] paddr;
    rand protection_type_e pprot;
    rand tx_type_e pwrite;
    rand transfer_size_e transfer_size;
    rand bit [DATA_WIDTH-1:0] pwdata;
    rand bit [(DATA_WIDTH/8)-1:0] pstrb;
    bit [DATA_WIDTH-1:0] prdata;
    slave_error_e pslverr;
    master_agent_config cfg_h;
    int no_of_wait_states_detected;

    function new(string name = "master_tx");
      super.new(name);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      master_tx rhs_tx;
      if (!$cast(rhs_tx, rhs)) begin
        `uvm_fatal("do_copy", "cast failed")
      end
      super.do_copy(rhs);
      paddr = rhs_tx.paddr;
      pprot = rhs_tx.pprot;
      pwrite = rhs_tx.pwrite;
      pwdata = rhs_tx.pwdata;
      pstrb = rhs_tx.pstrb;
      prdata = rhs_tx.prdata;
      pslverr = rhs_tx.pslverr;
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      master_tx rhs_tx;
      if (!$cast(rhs_tx, rhs)) begin
        `uvm_fatal("do_compare", "cast failed")
        return 0;
      end
      return super.do_compare(rhs_tx, comparer) &&
             paddr == rhs_tx.paddr &&
             pprot == rhs_tx.pprot &&
             pwrite == rhs_tx.pwrite &&
             pwdata == rhs_tx.pwdata &&
             pstrb == rhs_tx.pstrb &&
             prdata == rhs_tx.prdata &&
             pslverr == rhs_tx.pslverr;
    endfunction

    virtual function void do_print(uvm_printer printer);
      printer.print_string("pwrite", pwrite.name());
      printer.print_field("paddr", paddr, $bits(paddr), UVM_HEX);
      printer.print_field("pwdata", pwdata, $bits(pwdata), UVM_HEX);
      printer.print_string("transfer_size", transfer_size.name());
      printer.print_field("pstrb", pstrb, 4, UVM_BIN);
      printer.print_string("pprot", pprot.name());
      printer.print_field("prdata", prdata, $bits(prdata), UVM_HEX);
      printer.print_string("pslverr", pslverr.name());
    endfunction

    constraint pstrb_c { $countones(pstrb) >= 1; }
  endclass

  // Sequencer class
  class master_sequencer extends uvm_sequencer #(master_tx);
    `uvm_component_utils(master_sequencer)

    master_agent_config cfg_h;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db #(master_agent_config)::get(this, "", "master_agent_config", cfg_h))
        `uvm_fatal("CONFIG", "Failed to get master_agent_config")
    endfunction
  endclass

  // Driver class
  class master_driver extends uvm_driver #(master_tx);
    `uvm_component_utils(master_driver)

    master_agent_config cfg_h;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db #(master_agent_config)::get(this, "", "master_agent_config", cfg_h))
        `uvm_fatal("CONFIG", "Failed to get config")
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        seq_item_port.get_next_item(req);
        `uvm_info("DRV", $sformatf("Driving paddr=0x%0h, pwdata=0x%0h", req.paddr, req.pwdata), UVM_MEDIUM)
        #10;
        seq_item_port.item_done();
      end
    endtask
  endclass

  // Monitor class
  class master_monitor extends uvm_monitor;
    `uvm_component_utils(master_monitor)

    uvm_analysis_port #(master_tx) ap;
    master_agent_config cfg_h;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      ap = new("ap", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      master_tx item;
      item = master_tx::type_id::create("item");
      `uvm_info("MON", "Monitoring transaction", UVM_HIGH)
      ap.write(item);
    endtask
  endclass

  // Coverage class (like APB AVIP's apb_master_coverage)
  class master_coverage extends uvm_subscriber #(master_tx);
    `uvm_component_utils(master_coverage)

    master_agent_config cfg_h;

    covergroup master_cg with function sample(master_agent_config cfg, master_tx packet);
      option.per_instance = 1;

      PADDR_CP: coverpoint cfg.paddr {
        bins ADDR_LOW = {[0:100]};
        bins ADDR_MID = {[101:200]};
      }

      PWRITE_CP: coverpoint tx_type_e'(packet.pwrite) {
        bins WRITE_OP = {WRITE};
        bins READ_OP = {READ};
      }

      PWDATA_CP: coverpoint packet.pwdata {
        bins DATA[] = {[0:255]};
      }
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      master_cg = new();
    endfunction

    virtual function void write(master_tx t);
      `uvm_info("COV", "Sampling coverage", UVM_HIGH)
      master_cg.sample(cfg_h, t);
    endfunction

    virtual function void report_phase(uvm_phase phase);
      `uvm_info(get_type_name(), $sformatf("Coverage = %0.2f%%", master_cg.get_coverage()), UVM_NONE)
    endfunction
  endclass

  // Agent class (like APB AVIP's apb_master_agent)
  class master_agent extends uvm_agent;
    `uvm_component_utils(master_agent)

    master_driver drv;
    master_monitor mon;
    master_sequencer seqr;
    master_coverage cov;
    master_agent_config cfg_h;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      if (!uvm_config_db #(master_agent_config)::get(this, "", "master_agent_config", cfg_h))
        `uvm_fatal("CONFIG", "Failed to get master_agent_config")

      is_active = cfg_h.is_active;

      if (is_active == UVM_ACTIVE) begin
        drv = master_driver::type_id::create("drv", this);
        seqr = master_sequencer::type_id::create("seqr", this);
      end

      mon = master_monitor::type_id::create("mon", this);

      if (cfg_h.has_coverage) begin
        cov = master_coverage::type_id::create("cov", this);
      end
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      if (is_active == UVM_ACTIVE) begin
        drv.seq_item_port.connect(seqr.seq_item_export);
      end
      if (cfg_h.has_coverage) begin
        mon.ap.connect(cov.analysis_export);
      end
    endfunction
  endclass

endpackage : master_pkg

// Environment package
package env_pkg;
  `include "uvm_macros.svh"
  import uvm_pkg::*;
  import global_pkg::*;
  import master_pkg::*;

  // Environment config
  class env_config extends uvm_object;
    `uvm_object_utils(env_config)

    int no_of_slaves = 1;
    bit has_scoreboard = 1;
    bit has_virtual_seqr = 1;
    master_agent_config master_cfg;

    function new(string name = "env_config");
      super.new(name);
    endfunction
  endclass

  // Scoreboard (like APB AVIP's apb_scoreboard)
  class scoreboard extends uvm_scoreboard;
    `uvm_component_utils(scoreboard)

    uvm_tlm_analysis_fifo #(master_tx) master_fifo;
    int tx_count = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      master_fifo = new("master_fifo", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      master_tx tx;
      super.run_phase(phase);
      forever begin
        master_fifo.get(tx);
        tx_count++;
        `uvm_info("SB", $sformatf("Got transaction %0d: paddr=0x%0h", tx_count, tx.paddr), UVM_HIGH)
      end
    endtask

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      `uvm_info("SB", $sformatf("Total transactions: %0d", tx_count), UVM_MEDIUM)
    endfunction
  endclass

  // Virtual sequencer
  class virtual_sequencer extends uvm_sequencer;
    `uvm_component_utils(virtual_sequencer)

    master_sequencer master_seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  // Environment class
  class env extends uvm_env;
    `uvm_component_utils(env)

    master_agent agent;
    scoreboard sb;
    virtual_sequencer vseqr;
    env_config cfg;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      if (!uvm_config_db #(env_config)::get(this, "", "env_config", cfg))
        `uvm_fatal("CONFIG", "Failed to get env_config")

      agent = master_agent::type_id::create("agent", this);

      if (cfg.has_scoreboard)
        sb = scoreboard::type_id::create("sb", this);

      if (cfg.has_virtual_seqr)
        vseqr = virtual_sequencer::type_id::create("vseqr", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      if (cfg.has_scoreboard)
        agent.mon.ap.connect(sb.master_fifo.analysis_export);
      if (cfg.has_virtual_seqr)
        vseqr.master_seqr = agent.seqr;
    endfunction
  endclass

endpackage : env_pkg

// Test package
package test_pkg;
  `include "uvm_macros.svh"
  import uvm_pkg::*;
  import global_pkg::*;
  import master_pkg::*;
  import env_pkg::*;

  // Base sequence
  class base_seq extends uvm_sequence #(master_tx);
    `uvm_object_utils(base_seq)

    `uvm_declare_p_sequencer(master_sequencer)

    function new(string name = "base_seq");
      super.new(name);
    endfunction

    virtual task body();
      if (!$cast(p_sequencer, m_sequencer)) begin
        `uvm_error(get_full_name(), "Virtual sequencer pointer cast failed")
      end
    endtask
  endclass

  // Write sequence
  class write_seq extends base_seq;
    `uvm_object_utils(write_seq)

    function new(string name = "write_seq");
      super.new(name);
    endfunction

    virtual task body();
      super.body();
      `uvm_info("SEQ", "Starting write sequence", UVM_MEDIUM)
      req = master_tx::type_id::create("req");
      start_item(req);
      if (!req.randomize() with { pwrite == WRITE; })
        `uvm_error("SEQ", "Randomization failed")
      finish_item(req);
    endtask
  endclass

  // Base test (like APB AVIP's apb_base_test)
  class base_test extends uvm_test;
    `uvm_component_utils(base_test)

    env test_env;
    env_config env_cfg;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      setup_env_config();
      test_env = env::type_id::create("env", this);
    endfunction

    virtual function void setup_env_config();
      env_cfg = env_config::type_id::create("env_cfg");
      env_cfg.no_of_slaves = NO_OF_SLAVES;
      env_cfg.has_scoreboard = 1;
      env_cfg.has_virtual_seqr = 1;

      // Setup master agent config
      env_cfg.master_cfg = master_agent_config::type_id::create("master_cfg");
      if (MASTER_AGENT_ACTIVE)
        env_cfg.master_cfg.is_active = UVM_ACTIVE;
      else
        env_cfg.master_cfg.is_active = UVM_PASSIVE;

      // Set configs in database
      uvm_config_db #(master_agent_config)::set(this, "*agent*", "master_agent_config", env_cfg.master_cfg);
      uvm_config_db #(env_config)::set(this, "*", "env_config", env_cfg);

      `uvm_info(get_type_name(), $sformatf("\nENV CONFIG\n%s", env_cfg.sprint()), UVM_LOW)
    endfunction

    virtual function void end_of_elaboration_phase(uvm_phase phase);
      super.end_of_elaboration_phase(phase);
      uvm_top.print_topology();
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      super.run_phase(phase);
      #10;
      phase.drop_objection(this);
    endtask
  endclass

  // Write test
  class write_test extends base_test;
    `uvm_component_utils(write_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      write_seq seq;
      phase.raise_objection(this);

      seq = write_seq::type_id::create("seq");
      seq.start(test_env.agent.seqr);

      phase.drop_objection(this);
    endtask
  endclass

endpackage : test_pkg

// Top module
module uvm_apb_style_test_top;
  import uvm_pkg::*;
  initial begin
    run_test("write_test");
  end
endmodule
