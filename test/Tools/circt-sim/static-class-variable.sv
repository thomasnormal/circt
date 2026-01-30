// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir | FileCheck %s
// REQUIRES: slang

// Test static class variables work correctly in simulation
// Static class variables are stored in LLVM globals and accessed via
// llhd.prb of unrealized_conversion_cast from llvm.mlir.addressof

class Singleton;
    static Singleton m_inst;
    int value;
    
    static function Singleton get();
        if (m_inst == null) begin
            m_inst = new;
            m_inst.value = 0;
        end
        return m_inst;
    endfunction
    
    function void set_value(int v);
        value = v;
    endfunction
    
    function int get_value();
        return value;
    endfunction
endclass

module TestStaticClassVariable;
    Singleton s1, s2;
    initial begin
        // First call creates the singleton
        s1 = Singleton::get();
        s1.set_value(42);
        
        // Second call returns the same instance
        s2 = Singleton::get();
        
        // Verify same instance
        // CHECK: Same instance: YES
        if (s1 == s2)
            $display("Same instance: YES");
        else
            $display("Same instance: NO");
        
        // Verify value preserved
        // CHECK: Value: 42
        $display("Value: %0d", s2.get_value());
        
        // Direct static access
        // CHECK: Direct access: 42
        $display("Direct access: %0d", Singleton::m_inst.value);
        
        // CHECK: Test passed
        $display("Test passed");
    end
endmodule
