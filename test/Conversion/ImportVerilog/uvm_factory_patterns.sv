// RUN: circt-verilog --ir-moore %s 2>&1
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// UVM Factory Pattern and Advanced Type System Tests
//===----------------------------------------------------------------------===//

/// Simplified UVM base classes for factory testing

class uvm_object;
    static string type_name = "uvm_object";

    virtual function uvm_object create(string name = "");
        uvm_object obj = new;
        return obj;
    endfunction

    virtual function string get_type_name();
        return type_name;
    endfunction
endclass

class uvm_component extends uvm_object;
    static string type_name = "uvm_component";
    string name;

    function new(string name = "");
        this.name = name;
    endfunction

    virtual function string get_type_name();
        return type_name;
    endfunction
endclass

/// Test factory-like pattern with type override
class uvm_test extends uvm_component;
    static string type_name = "uvm_test";

    function new(string name = "uvm_test");
        super.new(name);
    endfunction

    virtual function void run_test();
        // Base test run
    endfunction
endclass

class my_test extends uvm_test;
    static string type_name = "my_test";

    function new(string name = "my_test");
        super.new(name);
    endfunction

    virtual function void run_test();
        super.run_test();
        // Custom test implementation
    endfunction
endclass

/// Test parameterized factory pattern
class typed_component #(type T = int);
    T data;
    static string type_id;

    static function typed_component#(T) create_object();
        typed_component#(T) obj = new;
        return obj;
    endfunction

    virtual function void set_data(T value);
        this.data = value;
    endfunction

    virtual function T get_data();
        return this.data;
    endfunction
endclass

/// Test registry pattern (UVM uses this)
class base_registry;
    static base_registry m_singleton;

    static function base_registry get();
        if (m_singleton == null)
            m_singleton = new;
        return m_singleton;
    endfunction

    virtual function uvm_object create_object(string name);
        uvm_object obj = new;
        return obj;
    endfunction
endclass

/// Test component hierarchy (common UVM pattern)
class uvm_env extends uvm_component;
    function new(string name = "env");
        super.new(name);
    endfunction

    virtual function void build_phase();
        // Build components
    endfunction
endclass

class my_env extends uvm_env;
    function new(string name = "my_env");
        super.new(name);
    endfunction

    virtual function void build_phase();
        // Create agents, scoreboards, etc.
    endfunction
endclass

/// Test phase methods (UVM phase system)
class phased_component extends uvm_component;
    function new(string name = "phased");
        super.new(name);
    endfunction

    virtual function void build_phase();
    endfunction

    virtual function void connect_phase();
    endfunction

    virtual task run_phase();
        // Run phase is a task
    endtask
endclass

/// Test transaction types
class base_transaction extends uvm_object;
    int addr;
    int data;

    function new(string name = "");
        // uvm_object doesn't have constructor
    endfunction

    virtual function uvm_object clone();
        base_transaction t = new;
        t.addr = this.addr;
        t.data = this.data;
        return t;
    endfunction

    virtual function bit compare(uvm_object rhs);
        base_transaction rhs_t;
        if (!$cast(rhs_t, rhs))
            return 0;
        return (this.addr == rhs_t.addr && this.data == rhs_t.data);
    endfunction
endclass

class extended_transaction extends base_transaction;
    bit [7:0] ctrl;

    function new(string name = "");
        // base_transaction constructor doesn't call super
    endfunction

    virtual function uvm_object clone();
        extended_transaction t = new;
        t.addr = this.addr;
        t.data = this.data;
        t.ctrl = this.ctrl;
        return t;
    endfunction
endclass

/// Test driver pattern (TLM)
class base_driver #(type REQ = base_transaction) extends uvm_component;
    function new(string name = "driver");
        super.new(name);
    endfunction

    virtual task drive_item(REQ item);
        // Drive transaction
    endtask
endclass

class my_driver extends base_driver#(extended_transaction);
    function new(string name = "my_driver");
        super.new(name);
    endfunction

    virtual task drive_item(extended_transaction item);
        // Drive extended transaction
    endtask
endclass

/// Test monitor pattern
class base_monitor extends uvm_component;
    function new(string name = "monitor");
        super.new(name);
    endfunction

    virtual task collect_transactions();
        // Monitor interface
    endtask
endclass

/// Test agent pattern (contains driver, monitor, sequencer)
class base_agent extends uvm_component;
    base_driver#(base_transaction) driver;
    base_monitor monitor;

    function new(string name = "agent");
        super.new(name);
    endfunction

    virtual function void build_phase();
        driver = new("driver");
        monitor = new("monitor");
    endfunction

    virtual function void connect_phase();
        // Connect driver and monitor
    endfunction
endclass

/// Test sequence item and sequence
class sequence_item extends base_transaction;
    rand bit [31:0] rand_data;

    function new(string name = "");
        // base_transaction constructor doesn't call super
    endfunction
endclass

class base_sequence extends uvm_object;
    function new(string name = "");
        // uvm_object doesn't have constructor with args
    endfunction

    virtual task body();
        // Sequence body
    endtask
endclass

/// Test callback mechanism (UVM callbacks)
class driver_callback;
    virtual function void pre_send(base_transaction tr);
        // Callback hook
    endfunction

    virtual function void post_send(base_transaction tr);
        // Callback hook
    endfunction
endclass

class my_callback extends driver_callback;
    virtual function void pre_send(base_transaction tr);
        // Custom pre-send behavior
    endfunction
endclass

/// Test config object pattern
class config_object extends uvm_object;
    bit has_coverage;
    bit has_scoreboard;
    int num_agents;

    function new(string name = "");
        has_coverage = 1;
        has_scoreboard = 1;
        num_agents = 1;
    endfunction
endclass

/// Test scoreboard pattern
class base_scoreboard extends uvm_component;
    int match_count;
    int mismatch_count;

    function new(string name = "scoreboard");
        super.new(name);
        match_count = 0;
        mismatch_count = 0;
    endfunction

    virtual function void compare_transaction(base_transaction actual, base_transaction expected);
        if (actual.compare(expected))
            match_count++;
        else
            mismatch_count++;
    endfunction

    virtual function void report();
        $display("Matches: %0d, Mismatches: %0d", match_count, mismatch_count);
    endfunction
endclass

/// Test full testbench hierarchy
class my_scoreboard extends base_scoreboard;
    function new(string name = "my_scoreboard");
        super.new(name);
    endfunction
endclass

class my_agent extends base_agent;
    my_driver my_drv;

    function new(string name = "my_agent");
        super.new(name);
    endfunction

    virtual function void build_phase();
        my_drv = new("my_driver");
        monitor = new("monitor");
    endfunction
endclass

class my_complete_env extends uvm_env;
    my_agent agent;
    my_scoreboard scoreboard;
    config_object cfg;

    function new(string name = "my_complete_env");
        super.new(name);
    endfunction

    virtual function void build_phase();
        super.build_phase();
        cfg = config_object::new();
        agent = my_agent::new("agent");
        if (cfg.has_scoreboard)
            scoreboard = my_scoreboard::new("scoreboard");
    endfunction
endclass

/// Test full test with everything
class my_complete_test extends uvm_test;
    my_complete_env env;

    function new(string name = "my_complete_test");
        super.new(name);
    endfunction

    virtual function void build_phase();
        env = my_complete_env::new("env");
    endfunction

    virtual function void run_test();
        super.run_test();
        // Run sequences, check results
    endfunction
endclass

/// Module to instantiate and test
module test_factory_patterns;
    my_complete_test test_inst;
    my_env env_inst;
    base_transaction trans;
    extended_transaction ext_trans;

    initial begin
        // Test basic object creation
        test_inst = new("test");

        // Test polymorphic assignment
        trans = new;
        ext_trans = new;
        trans = ext_trans;  // Upcast

        // Test virtual method call through base handle
        test_inst.run_test();

        // Test clone pattern
        // trans = trans.clone();  // Returns uvm_object, needs cast
    end
endmodule

/// Test type system with deep hierarchy
class level1_component extends uvm_component;
    virtual function void phase1();
    endfunction
endclass

class level2_component extends level1_component;
    virtual function void phase1();
        super.phase1();
    endfunction
    virtual function void phase2();
    endfunction
endclass

class level3_component extends level2_component;
    virtual function void phase1();
        super.phase1();
    endfunction
    virtual function void phase2();
        super.phase2();
    endfunction
    virtual function void phase3();
    endfunction
endclass

/// Test interface-like pattern (pure virtual)
virtual class base_interface;
    pure virtual function void send();
    pure virtual function bit recv();
endclass

class concrete_interface extends base_interface;
    virtual function void send();
        // Implementation
    endfunction

    virtual function bit recv();
        return 1;
    endfunction
endclass
