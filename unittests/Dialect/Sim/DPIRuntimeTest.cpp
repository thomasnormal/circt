//===- DPIRuntimeTest.cpp - Unit tests for DPIRuntime ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains unit tests for the DPIRuntime infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/DPIRuntime.h"
#include "gtest/gtest.h"

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// DPIValue Tests
//===----------------------------------------------------------------------===//

TEST(DPIValueTest, DefaultConstruction) {
  DPIValue val;
  EXPECT_EQ(val.getType(), DPIDataType::Void);
}

TEST(DPIValueTest, IntegerTypes) {
  DPIValue byteVal(static_cast<int8_t>(42));
  EXPECT_EQ(byteVal.getType(), DPIDataType::Byte);
  EXPECT_EQ(byteVal.getByte(), 42);

  DPIValue shortVal(static_cast<int16_t>(1234));
  EXPECT_EQ(shortVal.getType(), DPIDataType::ShortInt);
  EXPECT_EQ(shortVal.getShortInt(), 1234);

  DPIValue intVal(static_cast<int32_t>(123456));
  EXPECT_EQ(intVal.getType(), DPIDataType::Int);
  EXPECT_EQ(intVal.getInt(), 123456);

  DPIValue longVal(static_cast<int64_t>(123456789012LL));
  EXPECT_EQ(longVal.getType(), DPIDataType::LongInt);
  EXPECT_EQ(longVal.getLongInt(), 123456789012LL);
}

TEST(DPIValueTest, FloatingPointTypes) {
  DPIValue floatVal(3.14f);
  EXPECT_EQ(floatVal.getType(), DPIDataType::ShortReal);
  EXPECT_FLOAT_EQ(floatVal.getShortReal(), 3.14f);

  DPIValue doubleVal(3.14159265358979);
  EXPECT_EQ(doubleVal.getType(), DPIDataType::Real);
  EXPECT_DOUBLE_EQ(doubleVal.getReal(), 3.14159265358979);
}

TEST(DPIValueTest, StringType) {
  DPIValue strVal("hello world");
  EXPECT_EQ(strVal.getType(), DPIDataType::String);
  EXPECT_EQ(strVal.getString(), "hello world");

  DPIValue strVal2(std::string("test string"));
  EXPECT_EQ(strVal2.getString(), "test string");
}

TEST(DPIValueTest, CHandleType) {
  int dummy = 42;
  DPIValue ptrVal(static_cast<void *>(&dummy));
  EXPECT_EQ(ptrVal.getType(), DPIDataType::CHandle);
  EXPECT_EQ(ptrVal.getCHandle(), &dummy);
}

TEST(DPIValueTest, VectorType) {
  std::vector<uint32_t> bits = {0xDEADBEEF, 0x12345678};
  DPIValue vecVal(bits, 64, false);

  EXPECT_EQ(vecVal.getType(), DPIDataType::BitVector);
  EXPECT_EQ(vecVal.getBitWidth(), 64u);
  EXPECT_EQ(vecVal.getVector().size(), 2u);
  EXPECT_EQ(vecVal.getVector()[0], 0xDEADBEEF);
}

TEST(DPIValueTest, Conversion) {
  DPIValue intVal(static_cast<int32_t>(42));
  EXPECT_EQ(intVal.toInt64(), 42);
  EXPECT_DOUBLE_EQ(intVal.toDouble(), 42.0);

  DPIValue realVal(3.14);
  EXPECT_DOUBLE_EQ(realVal.toDouble(), 3.14);
}

//===----------------------------------------------------------------------===//
// DPIArgument Tests
//===----------------------------------------------------------------------===//

TEST(DPIArgumentTest, Construction) {
  DPIArgument arg("input_data", DPIDataType::Int, DPIArgDirection::Input, 32);

  EXPECT_EQ(arg.name, "input_data");
  EXPECT_EQ(arg.dataType, DPIDataType::Int);
  EXPECT_EQ(arg.direction, DPIArgDirection::Input);
  EXPECT_EQ(arg.bitWidth, 32u);
  EXPECT_TRUE(arg.isInput());
  EXPECT_FALSE(arg.isOutput());
}

TEST(DPIArgumentTest, InOutDirection) {
  DPIArgument arg("bidir", DPIDataType::Int, DPIArgDirection::InOut, 32);

  EXPECT_TRUE(arg.isInput());
  EXPECT_TRUE(arg.isOutput());
}

//===----------------------------------------------------------------------===//
// DPIFunctionSignature Tests
//===----------------------------------------------------------------------===//

TEST(DPIFunctionSignatureTest, Construction) {
  DPIFunctionSignature sig("my_func", "c_my_func");

  EXPECT_EQ(sig.name, "my_func");
  EXPECT_EQ(sig.cName, "c_my_func");
  EXPECT_EQ(sig.returnType, DPIDataType::Void);
}

TEST(DPIFunctionSignatureTest, AddArguments) {
  DPIFunctionSignature sig("add");
  sig.returnType = DPIDataType::Int;
  sig.addInput("a", DPIDataType::Int);
  sig.addInput("b", DPIDataType::Int);

  EXPECT_EQ(sig.arguments.size(), 2u);
  EXPECT_EQ(sig.getInputCount(), 2u);
  EXPECT_EQ(sig.getOutputCount(), 0u);
}

TEST(DPIFunctionSignatureTest, MixedArguments) {
  DPIFunctionSignature sig("process");
  sig.addInput("in", DPIDataType::Int);
  sig.addOutput("out", DPIDataType::Int);

  EXPECT_EQ(sig.arguments.size(), 2u);
  EXPECT_EQ(sig.getInputCount(), 1u);
  EXPECT_EQ(sig.getOutputCount(), 1u);
}

//===----------------------------------------------------------------------===//
// DPIContext Tests
//===----------------------------------------------------------------------===//

TEST(DPIContextTest, TimeManagement) {
  DPIContext ctx;

  SimTime t(1000, 5, 2);
  ctx.setTime(t);

  EXPECT_EQ(ctx.getTime().realTime, 1000u);
  EXPECT_EQ(ctx.getTime().deltaStep, 5u);
}

TEST(DPIContextTest, ScopeManagement) {
  DPIContext ctx;

  ctx.setScope("top.cpu.alu");
  EXPECT_EQ(ctx.getScope(), "top.cpu.alu");

  ctx.setInstanceName("alu_inst");
  EXPECT_EQ(ctx.getInstanceName(), "alu_inst");
}

TEST(DPIContextTest, LocalVariables) {
  DPIContext ctx;

  ctx.setLocal("counter", DPIValue(static_cast<int32_t>(42)));

  auto *val = ctx.getLocal("counter");
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(val->getInt(), 42);

  auto *missing = ctx.getLocal("nonexistent");
  EXPECT_EQ(missing, nullptr);
}

//===----------------------------------------------------------------------===//
// DPIRuntime Tests
//===----------------------------------------------------------------------===//

TEST(DPIRuntimeTest, Construction) {
  DPIRuntime::Config config;
  config.debug = false;
  config.typeCheck = true;

  DPIRuntime runtime(config);

  EXPECT_EQ(runtime.getStatistics().functionsRegistered, 0u);
  EXPECT_EQ(runtime.getStatistics().librariesLoaded, 0u);
}

TEST(DPIRuntimeTest, RegisterImport) {
  DPIRuntime runtime;

  DPIFunctionSignature sig("test_func");
  sig.returnType = DPIDataType::Int;
  sig.addInput("x", DPIDataType::Int);

  runtime.registerImport(sig, [](const std::vector<DPIValue> &args) {
    return DPIValue(static_cast<int32_t>(args[0].getInt() * 2));
  });

  EXPECT_TRUE(runtime.hasFunction("test_func"));
  EXPECT_EQ(runtime.getStatistics().functionsRegistered, 1u);
}

TEST(DPIRuntimeTest, CallFunction) {
  DPIRuntime runtime;

  DPIFunctionSignature sig("double_it");
  sig.returnType = DPIDataType::Int;
  sig.addInput("x", DPIDataType::Int);

  runtime.registerImport(sig, [](const std::vector<DPIValue> &args) {
    return DPIValue(static_cast<int32_t>(args[0].getInt() * 2));
  });

  std::vector<DPIValue> args;
  args.emplace_back(static_cast<int32_t>(21));

  DPIValue result = runtime.call("double_it", args);

  EXPECT_EQ(result.getInt(), 42);
  EXPECT_EQ(runtime.getStatistics().callsMade, 1u);
}

TEST(DPIRuntimeTest, CallWithContext) {
  DPIRuntime runtime;

  DPIFunctionSignature sig("get_time");
  sig.returnType = DPIDataType::LongInt;
  sig.isContext = true;

  runtime.registerImport(sig, [&runtime](const std::vector<DPIValue> &) {
    const auto &ctx = runtime.getCurrentContext();
    return DPIValue(static_cast<int64_t>(ctx.getTime().realTime));
  });

  DPIContext ctx;
  ctx.setTime(SimTime(12345, 0, 0));

  DPIValue result = runtime.callWithContext("get_time", {}, ctx);

  EXPECT_EQ(result.getLongInt(), 12345);
}

TEST(DPIRuntimeTest, ContextStack) {
  DPIRuntime runtime;

  DPIContext ctx1;
  ctx1.setScope("top.a");

  DPIContext ctx2;
  ctx2.setScope("top.b");

  EXPECT_EQ(runtime.getCurrentContext().getScope(), "");

  runtime.pushContext(ctx1);
  EXPECT_EQ(runtime.getCurrentContext().getScope(), "top.a");

  runtime.pushContext(ctx2);
  EXPECT_EQ(runtime.getCurrentContext().getScope(), "top.b");

  runtime.popContext();
  EXPECT_EQ(runtime.getCurrentContext().getScope(), "top.a");

  runtime.popContext();
  EXPECT_EQ(runtime.getCurrentContext().getScope(), "");
}

TEST(DPIRuntimeTest, RegisterExport) {
  DPIRuntime runtime;

  DPIFunctionSignature sig("sv_callback");
  sig.returnType = DPIDataType::Void;

  bool called = false;
  runtime.registerExport(sig, [&called](const std::vector<DPIValue> &) {
    called = true;
    return DPIValue();
  });

  EXPECT_TRUE(runtime.hasFunction("sv_callback"));

  const auto *reg = runtime.getFunction("sv_callback");
  ASSERT_NE(reg, nullptr);
  EXPECT_FALSE(reg->isImport);
}

TEST(DPIRuntimeTest, TypeCheckFailure) {
  DPIRuntime::Config config;
  config.typeCheck = true;
  config.debug = false;
  DPIRuntime runtime(config);

  DPIFunctionSignature sig("typed_func");
  sig.returnType = DPIDataType::Int;
  sig.addInput("a", DPIDataType::Int);
  sig.addInput("b", DPIDataType::Int);

  runtime.registerImport(sig, [](const std::vector<DPIValue> &args) {
    return DPIValue(static_cast<int32_t>(args[0].getInt() + args[1].getInt()));
  });

  // Wrong number of arguments
  std::vector<DPIValue> wrongArgs;
  wrongArgs.emplace_back(static_cast<int32_t>(1));

  DPIValue result = runtime.call("typed_func", wrongArgs);

  // Should fail and increment error count
  EXPECT_EQ(runtime.getStatistics().callErrors, 1u);
}

TEST(DPIRuntimeTest, UnknownFunction) {
  DPIRuntime runtime;

  DPIValue result = runtime.call("nonexistent", {});

  EXPECT_EQ(runtime.getStatistics().callErrors, 1u);
}

//===----------------------------------------------------------------------===//
// DPIRuntime Utility Tests
//===----------------------------------------------------------------------===//

TEST(DPIRuntimeUtilityTest, IntWidthToDPIType) {
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(1), DPIDataType::Byte);
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(8), DPIDataType::Byte);
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(16), DPIDataType::ShortInt);
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(32), DPIDataType::Int);
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(64), DPIDataType::LongInt);
  EXPECT_EQ(DPIRuntime::intWidthToDPIType(128), DPIDataType::BitVector);
}

TEST(DPIRuntimeUtilityTest, GetCPointerType) {
  EXPECT_EQ(DPIRuntime::getCPointerType(DPIDataType::Byte), "int8_t*");
  EXPECT_EQ(DPIRuntime::getCPointerType(DPIDataType::Int), "int32_t*");
  EXPECT_EQ(DPIRuntime::getCPointerType(DPIDataType::Real), "double*");
  EXPECT_EQ(DPIRuntime::getCPointerType(DPIDataType::String), "const char**");
}

TEST(DPIRuntimeUtilityTest, GenerateCDeclaration) {
  DPIFunctionSignature sig("add_values");
  sig.cName = "c_add_values";
  sig.returnType = DPIDataType::Int;
  sig.addInput("a", DPIDataType::Int);
  sig.addInput("b", DPIDataType::Int);

  std::string decl = DPIRuntime::generateCDeclaration(sig);

  EXPECT_TRUE(decl.find("int32_t c_add_values") != std::string::npos);
  EXPECT_TRUE(decl.find("int32_t a") != std::string::npos);
  EXPECT_TRUE(decl.find("int32_t b") != std::string::npos);
}

TEST(DPIRuntimeUtilityTest, GenerateCDeclarationWithOutputs) {
  DPIFunctionSignature sig("compute");
  sig.cName = "c_compute";
  sig.returnType = DPIDataType::Void;
  sig.addInput("in", DPIDataType::Int);
  sig.addOutput("out", DPIDataType::Int);

  std::string decl = DPIRuntime::generateCDeclaration(sig);

  EXPECT_TRUE(decl.find("void c_compute") != std::string::npos);
  EXPECT_TRUE(decl.find("int32_t in") != std::string::npos);
  EXPECT_TRUE(decl.find("int32_t* out") != std::string::npos);
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST(DPIRuntimeIntegration, CompleteCallSequence) {
  DPIRuntime runtime;

  // Register a simple ALU function
  DPIFunctionSignature aluSig("alu");
  aluSig.returnType = DPIDataType::Int;
  aluSig.addInput("a", DPIDataType::Int);
  aluSig.addInput("b", DPIDataType::Int);
  aluSig.addInput("op", DPIDataType::Byte);

  runtime.registerImport(aluSig, [](const std::vector<DPIValue> &args) {
    int32_t a = args[0].getInt();
    int32_t b = args[1].getInt();
    int8_t op = args[2].getByte();

    int32_t result;
    switch (op) {
    case 0:
      result = a + b;
      break;
    case 1:
      result = a - b;
      break;
    case 2:
      result = a & b;
      break;
    case 3:
      result = a | b;
      break;
    default:
      result = 0;
    }
    return DPIValue(result);
  });

  // Test addition
  std::vector<DPIValue> addArgs;
  addArgs.emplace_back(static_cast<int32_t>(10));
  addArgs.emplace_back(static_cast<int32_t>(5));
  addArgs.emplace_back(static_cast<int8_t>(0));

  DPIValue addResult = runtime.call("alu", addArgs);
  EXPECT_EQ(addResult.getInt(), 15);

  // Test subtraction
  std::vector<DPIValue> subArgs;
  subArgs.emplace_back(static_cast<int32_t>(10));
  subArgs.emplace_back(static_cast<int32_t>(5));
  subArgs.emplace_back(static_cast<int8_t>(1));

  DPIValue subResult = runtime.call("alu", subArgs);
  EXPECT_EQ(subResult.getInt(), 5);

  // Check statistics
  EXPECT_EQ(runtime.getStatistics().callsMade, 2u);
  EXPECT_EQ(runtime.getStatistics().callErrors, 0u);
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
