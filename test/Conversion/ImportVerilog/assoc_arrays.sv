// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Associative Array Tests for UVM Parsing Features
//===----------------------------------------------------------------------===//

/// Test associative array declaration with different key types
// CHECK-LABEL: moore.module @AssocArrayDeclarationTest() {
module AssocArrayDeclarationTest;
    // CHECK: [[AA1:%.+]] = moore.variable : <!moore.assoc_array<i32, i32>>
    int aa_int[int];
    // CHECK: [[AA2:%.+]] = moore.variable : <!moore.assoc_array<i32, string>>
    int aa_string[string];
    // CHECK: [[AA3:%.+]] = moore.variable : <!moore.assoc_array<l8, i32>>
    logic [7:0] aa_logic[int];
endmodule

/// Test associative array exists() method
/// exists() returns 1 if the key exists, 0 otherwise
// CHECK-LABEL: moore.module @AssocArrayExistsTest() {
module AssocArrayExistsTest;
    int aa[int];
    int result;

    initial begin
        aa[10] = 100;
        aa[20] = 200;

        // Check if key exists
        // CHECK: moore.assoc.exists
        result = aa.exists(10);

        // Check if key does not exist
        // CHECK: moore.assoc.exists
        result = aa.exists(30);
    end
endmodule

/// Test associative array delete() method with key
/// delete(key) removes the entry with the specified key
// CHECK-LABEL: moore.module @AssocArrayDeleteTest() {
module AssocArrayDeleteTest;
    int aa[int];

    initial begin
        aa[10] = 100;
        aa[20] = 200;
        aa[30] = 300;

        // Delete specific key
        // CHECK: moore.assoc.delete_key %aa, {{%.+}} : <!moore.assoc_array<i32, i32>>, i32
        aa.delete(20);

        // Delete all entries
        // CHECK: moore.assoc.delete %aa : <!moore.assoc_array<i32, i32>>
        aa.delete();
    end
endmodule

// TODO: Enable when first() is implemented as a subroutine
// /// Test associative array first() method
// /// first(ref key) sets key to the first (smallest) key and returns 1 if array is non-empty
// // CHECK-LABEL: moore.module @AssocArrayFirstTest() {
// module AssocArrayFirstTest;
//     int aa[int];
//     int key;
//     int found;
//
//     initial begin
//         aa[30] = 300;
//         aa[10] = 100;
//         aa[20] = 200;
//
//         // Get first key (should be 10 - smallest)
//         // CHECK: moore.assoc.first
//         found = aa.first(key);
//     end
// endmodule

// TODO: Enable when next() is implemented as a subroutine
// /// Test associative array next() method
// /// next(ref key) sets key to the next key after current key value
// // CHECK-LABEL: moore.module @AssocArrayNextTest() {
// module AssocArrayNextTest;
//     int aa[int];
//     int key;
//     int found;
//
//     initial begin
//         aa[10] = 100;
//         aa[20] = 200;
//         aa[30] = 300;
//
//         // Get first key
//         // CHECK: moore.assoc.first
//         found = aa.first(key);
//
//         // Get next key
//         // CHECK: moore.assoc.next
//         found = aa.next(key);
//
//         // Get next key again
//         // CHECK: moore.assoc.next
//         found = aa.next(key);
//     end
// endmodule

// TODO: Enable when prev() is implemented as a subroutine
// /// Test associative array prev() method
// /// prev(ref key) sets key to the previous key before current key value
// // CHECK-LABEL: moore.module @AssocArrayPrevTest() {
// module AssocArrayPrevTest;
//     int aa[int];
//     int key;
//     int found;
//
//     initial begin
//         aa[10] = 100;
//         aa[20] = 200;
//         aa[30] = 300;
//
//         // Start from a key
//         key = 30;
//
//         // Get previous key (should be 20)
//         // Note: prev may emit a warning as it's not fully supported
//         found = aa.prev(key);
//     end
// endmodule

// TODO: Enable when first()/next() are implemented as subroutines
// /// Test associative array iteration pattern (UVM common pattern)
// // CHECK-LABEL: moore.module @AssocArrayIterationTest() {
// module AssocArrayIterationTest;
//     int aa[int];
//     int key;
//     int value;
//
//     initial begin
//         // Populate array
//         aa[100] = 1;
//         aa[200] = 2;
//         aa[300] = 3;
//
//         // Iterate through all keys using first/next pattern
//         // CHECK: moore.assoc.first
//         if (aa.first(key)) begin
//             // CHECK: moore.dyn_extract
//             value = aa[key];
//             // CHECK: moore.assoc.next
//             while (aa.next(key)) begin
//                 // CHECK: moore.dyn_extract
//                 value = aa[key];
//             end
//         end
//     end
// endmodule

/// Test associative array with string keys (common in UVM)
// CHECK-LABEL: moore.module @AssocArrayStringKeyTest() {
module AssocArrayStringKeyTest;
    // CHECK: moore.variable : <!moore.assoc_array<i32, string>>
    int cfg_data[string];
    int val;

    initial begin
        // Set values with string keys
        cfg_data["width"] = 32;
        cfg_data["depth"] = 64;
        cfg_data["enable"] = 1;

        // Read values
        // CHECK: moore.dyn_extract
        val = cfg_data["width"];
    end
endmodule

/// Test associative array element access and assignment
// CHECK-LABEL: moore.module @AssocArrayAccessTest() {
module AssocArrayAccessTest;
    int aa[int];
    int val;

    initial begin
        // Write to associative array
        // CHECK: moore.dyn_extract_ref
        // CHECK: moore.blocking_assign
        aa[5] = 50;
        aa[10] = 100;
        aa[15] = 150;

        // Read from associative array
        // CHECK: moore.dyn_extract
        val = aa[10];

        // Modify existing entry
        // CHECK: moore.dyn_extract_ref
        // CHECK: moore.blocking_assign
        aa[10] = 200;
    end
endmodule

/// Test associative array num() method (alias for size())
// CHECK-LABEL: moore.module @AssocArrayNumTest() {
module AssocArrayNumTest;
    int aa[int];
    int count;

    initial begin
        aa[1] = 10;
        aa[2] = 20;
        aa[3] = 30;

        // Get number of entries
        // CHECK: moore.array.size
        count = aa.size();
    end
endmodule

//===----------------------------------------------------------------------===//
// Associative Array in Class Context (UVM Pattern)
//===----------------------------------------------------------------------===//

/// Test associative array within a class (UVM resource pool pattern)
// CHECK-LABEL: moore.class.classdecl @ResourcePool {
// CHECK:   moore.class.propertydecl @resources : !moore.assoc_array<i32, string>
// CHECK: }
class ResourcePool;
    int resources[string];

    function void set_resource(string name, int value);
        resources[name] = value;
    endfunction

    function int get_resource(string name);
        if (resources.exists(name))
            return resources[name];
        return -1;
    endfunction

    function bit has_resource(string name);
        return resources.exists(name);
    endfunction
endclass

// CHECK-LABEL: moore.module @AssocArrayInClassTest() {
module AssocArrayInClassTest;
    ResourcePool pool;

    initial begin
        pool = new;
        pool.set_resource("timeout", 1000);
        pool.set_resource("retries", 3);
    end
endmodule

//===----------------------------------------------------------------------===//
// Combined Queue and Associative Array Tests
//===----------------------------------------------------------------------===//

/// Test using both queues and associative arrays together
// CHECK-LABEL: moore.module @QueueAndAssocArrayTest() {
module QueueAndAssocArrayTest;
    int queue_data[$];
    int aa_data[int];
    int key;
    int val;

    initial begin
        // Populate queue
        queue_data.push_back(1);
        queue_data.push_back(2);
        queue_data.push_back(3);

        // Populate associative array using queue indices
        for (int i = 0; i < queue_data.size(); i++) begin
            // CHECK: moore.dyn_extract_ref
            aa_data[i] = queue_data[i] * 10;
        end

        // Read back from associative array
        // CHECK: moore.dyn_extract
        val = aa_data[0];
    end
endmodule

//===----------------------------------------------------------------------===//
// Typedef'd Associative Array Tests
//===----------------------------------------------------------------------===//

/// Test element select into typedef'd associative arrays (UVM pattern)
/// This tests the fix for "unsupported expression: element select into TypedefArray"
// CHECK-LABEL: moore.module @TypedefAssocArrayTest() {
typedef int assoc_int_typedef[int];
typedef int assoc_string_typedef[string];

module TypedefAssocArrayTest;
    // CHECK: moore.variable : <!moore.assoc_array<i32, i32>>
    assoc_int_typedef aa_int;
    // CHECK: moore.variable : <!moore.assoc_array<i32, string>>
    assoc_string_typedef aa_string;
    int val;

    initial begin
        // Write to typedef'd associative array with int key
        // CHECK: moore.dyn_extract_ref
        aa_int[5] = 50;

        // Read from typedef'd associative array with int key
        // CHECK: moore.dyn_extract
        val = aa_int[5];

        // Write to typedef'd associative array with string key
        // CHECK: moore.dyn_extract_ref
        aa_string["hello"] = 42;

        // Read from typedef'd associative array with string key
        // CHECK: moore.dyn_extract
        val = aa_string["hello"];
    end
endmodule

/// Test UVM-like severity_id_verbosities pattern with typedef'd key type
// CHECK-LABEL: moore.module @UVMVerbositiesPatternTest() {
typedef string uvm_id_verbosities_array;
typedef int severity_id_verbosities_typedef[uvm_id_verbosities_array];

module UVMVerbositiesPatternTest;
    // CHECK: moore.variable : <!moore.assoc_array<i32, string>>
    severity_id_verbosities_typedef verbosities;
    string severity;
    int level;

    initial begin
        severity = "UVM_LOW";

        // Write verbosity level with variable key
        // CHECK: moore.dyn_extract_ref
        verbosities[severity] = 100;

        // Read verbosity level with variable key
        // CHECK: moore.dyn_extract
        level = verbosities[severity];

        // Access with literal string key
        // CHECK: moore.dyn_extract_ref
        verbosities["UVM_HIGH"] = 500;

        // CHECK: moore.dyn_extract
        level = verbosities["UVM_HIGH"];
    end
endmodule
