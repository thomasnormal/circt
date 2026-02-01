// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Queue Operation Tests for UVM Parsing Features
//===----------------------------------------------------------------------===//

/// Test queue declaration with unbounded size
// CHECK-LABEL: moore.module @QueueDeclarationTest() {
module QueueDeclarationTest;
    // CHECK: [[Q:%.+]] = moore.variable : <queue<i32, 0>>
    int q[$];
    // CHECK: [[Q2:%.+]] = moore.variable : <queue<l8, 0>>
    logic [7:0] q2[$];
    // CHECK: [[Q3:%.+]] = moore.variable : <queue<i32, 10>>
    int q3[$:10];  // bounded queue
endmodule

/// Test queue push_back and push_front methods
// CHECK-LABEL: moore.module @QueuePushTest() {
module QueuePushTest;
    int q[$];

    initial begin
        // CHECK: moore.queue.push_back
        q.push_back(1);
        // CHECK: moore.queue.push_back
        q.push_back(2);
        // CHECK: moore.queue.push_front
        q.push_front(0);
    end
endmodule

/// Test queue insert method
// CHECK-LABEL: moore.module @QueueInsertTest() {
module QueueInsertTest;
    int q[$];

    initial begin
        q.push_back(1);
        q.push_back(3);
        // CHECK: moore.queue.insert
        q.insert(1, 2);
    end
endmodule

/// Test queue pop_back and pop_front methods
// CHECK-LABEL: moore.module @QueuePopTest() {
module QueuePopTest;
    int q[$];
    int val;

    initial begin
        q.push_back(1);
        q.push_back(2);
        q.push_back(3);
        // CHECK: moore.queue.pop_back
        val = q.pop_back();
        // CHECK: moore.queue.pop_front
        val = q.pop_front();
    end
endmodule

/// Test queue size method
// CHECK-LABEL: moore.module @QueueSizeTest() {
module QueueSizeTest;
    int q[$];
    int sz;

    initial begin
        q.push_back(1);
        q.push_back(2);
        // CHECK: moore.array.size
        sz = q.size();
    end
endmodule

/// Test queue delete method
// CHECK-LABEL: moore.module @QueueDeleteTest() {
module QueueDeleteTest;
    int q[$];

    initial begin
        q.push_back(1);
        q.push_back(2);
        // CHECK: moore.queue.delete
        q.delete();
    end
endmodule

/// Test queue element access with $ unbounded literal
/// The $ symbol represents the last element index (size - 1)
// CHECK-LABEL: moore.module @QueueDollarAccessTest() {
module QueueDollarAccessTest;
    int q[$];
    int first_elem;
    int last_elem;

    initial begin
        q.push_back(10);
        q.push_back(20);
        q.push_back(30);

        // Access first element
        // CHECK: moore.dyn_extract
        first_elem = q[0];

        // Access last element using $ - this represents q[q.size()-1]
        // The $ literal should be converted to (size - 1)
        // CHECK: moore.array.size
        // CHECK: moore.sub
        // CHECK: moore.dyn_extract
        last_elem = q[$];
    end
endmodule

/// Test queue slice with $ unbounded literal
// CHECK-LABEL: moore.module @QueueDollarSliceTest() {
module QueueDollarSliceTest;
    int q[$];
    int slice1[$];
    int slice2[$];

    initial begin
        q.push_back(1);
        q.push_back(2);
        q.push_back(3);
        q.push_back(4);

        // slice using $ as end index
        // CHECK: moore.array.size
        // CHECK: moore.sub
        // CHECK: moore.queue.slice
        slice1 = q[1:$];

        // slice using $ in arithmetic expression
        // CHECK: moore.array.size
        // CHECK: moore.sub
        // CHECK: moore.queue.slice
        slice2 = q[0:$-1];
    end
endmodule

/// Test queue with different element types
// CHECK-LABEL: moore.module @QueueTypesTest() {
module QueueTypesTest;
    // CHECK: moore.variable : <queue<i8, 0>>
    byte byte_q[$];
    // CHECK: moore.variable : <queue<i64, 0>>
    longint long_q[$];
    // CHECK: moore.variable : <queue<string, 0>>
    string str_q[$];
    // CHECK: moore.variable : <queue<l32, 0>>
    logic [31:0] logic_q[$];

    initial begin
        byte_q.push_back(8'hAB);
        long_q.push_back(64'd12345678);
        str_q.push_back("hello");
        logic_q.push_back(32'hDEADBEEF);
    end
endmodule

//===----------------------------------------------------------------------===//
// String Method Tests
//===----------------------------------------------------------------------===//

// Note: String itoa() method tests are commented out due to a known limitation
// where slang promotes integer arguments to logic types, but moore.int_to_string
// requires two-valued integer types. The itoa method IS recognized by the
// converter but needs a type conversion fix.
//
// TODO: Add itoa tests once the type conversion issue is resolved:
// - s.itoa(num) should convert integer to decimal string
// - Works with int, byte, shortint, longint types

/// Test string methods used together (UVM pattern)
// CHECK-LABEL: moore.module @StringMethodsComboTest() {
module StringMethodsComboTest;
    string s;
    string result;
    int len;

    initial begin
        s = "Hello";
        // CHECK: moore.string.len
        len = s.len();
        // CHECK: moore.string.toupper
        result = s.toupper();
        // CHECK: moore.string.tolower
        result = s.tolower();
    end
endmodule

//===----------------------------------------------------------------------===//
// Queue in Class Context (UVM Pattern)
//===----------------------------------------------------------------------===//

/// Test queue usage within a class (common UVM pattern)
// CHECK-LABEL: moore.class.classdecl @QueueContainer {
// CHECK:   moore.class.propertydecl @items : !moore.queue<i32, 0>
// CHECK: }
class QueueContainer;
    int items[$];

    function void add_item(int item);
        items.push_back(item);
    endfunction

    function int get_last();
        return items[$];
    endfunction

    function int get_size();
        return items.size();
    endfunction
endclass

// CHECK-LABEL: moore.module @QueueInClassTest() {
module QueueInClassTest;
    QueueContainer container;

    initial begin
        container = new;
        container.add_item(1);
        container.add_item(2);
        container.add_item(3);
    end
endmodule

//===----------------------------------------------------------------------===//
// Queue Concatenation Tests (UVM Pattern)
//===----------------------------------------------------------------------===//

/// Test queue concatenation with curly braces
/// This is the pattern used in UVM: all_callbacks = { all_callbacks, unique_callbacks };
// CHECK-LABEL: moore.module @QueueConcatTest() {
module QueueConcatTest;
    int q1[$];
    int q2[$];
    int result[$];

    initial begin
        q1.push_back(1);
        q1.push_back(2);
        q2.push_back(3);
        q2.push_back(4);
        // CHECK: [[Q1:%.+]] = moore.read %q1 : <queue<i32, 0>>
        // CHECK: [[Q2:%.+]] = moore.read %q2 : <queue<i32, 0>>
        // CHECK: [[CONCAT:%.+]] = moore.queue.concat([[Q1]], [[Q2]]) : !moore.queue<i32, 0>, !moore.queue<i32, 0> -> <i32, 0>
        // CHECK: moore.blocking_assign %result, [[CONCAT]] : queue<i32, 0>
        result = { q1, q2 };
    end
endmodule

//===----------------------------------------------------------------------===//
// Queue Concatenation With Elements
//===----------------------------------------------------------------------===//

/// Test queue concatenation with single element operands
// CHECK-LABEL: moore.module @QueueConcatElementTest() {
module QueueConcatElementTest;
    int q[$];
    int result[$];

    initial begin
        // CHECK: [[Q:%.+]] = moore.read %q : <queue<i32, 0>>
        // CHECK: moore.queue.concat
        // CHECK: moore.queue.push_back
        // CHECK: moore.queue.concat
        result = { q, 5 };

        // CHECK: moore.queue.concat
        // CHECK: moore.queue.push_back
        // CHECK: [[Q2:%.+]] = moore.read %q : <queue<i32, 0>>
        // CHECK: moore.queue.concat
        result = { 6, q };
    end
endmodule

//===----------------------------------------------------------------------===//
// Streaming Concatenation with Queue (UVM Pattern)
//===----------------------------------------------------------------------===//

/// Test streaming concatenation with string queue - concatenates all strings
// CHECK-LABEL: moore.module @StreamingConcatWithQueue() {
module StreamingConcatWithQueue;
    string queue[$];
    string result;

    initial begin
        queue.push_back("hello");
        queue.push_back("world");
        // CHECK: [[Q:%.+]] = moore.read %queue : <queue<string, 0>>
        // CHECK: [[STR:%.+]] = moore.stream_concat [[Q]] : queue<string, 0> -> string
        // CHECK: moore.blocking_assign %result, [[STR]] : string
        result = {>>{queue}};
    end
endmodule
