//===- WaveformDumperTest.cpp - Unit tests for WaveformDumper -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains unit tests for the WaveformDumper infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/WaveformDumper.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// TracedSignal Tests
//===----------------------------------------------------------------------===//

TEST(TracedSignalTest, Construction) {
  TracedSignal sig(1, "top.counter", "counter", 8, SignalType::Reg);

  EXPECT_EQ(sig.signalId, 1u);
  EXPECT_EQ(sig.path, "top.counter");
  EXPECT_EQ(sig.name, "counter");
  EXPECT_EQ(sig.width, 8u);
  EXPECT_EQ(sig.type, SignalType::Reg);
  EXPECT_TRUE(sig.isUnknown);
  EXPECT_EQ(sig.currentValue, 0u);
}

TEST(TracedSignalTest, ArrayIndex) {
  TracedSignal sig(1, "top.mem", "mem", 32, SignalType::Wire);
  sig.arrayIndex = 5;

  EXPECT_EQ(sig.arrayIndex, 5);
}

//===----------------------------------------------------------------------===//
// WaveformScope Tests
//===----------------------------------------------------------------------===//

TEST(WaveformScopeTest, HierarchicalPaths) {
  WaveformScope root("", "");
  auto *top = root.findOrCreateChild("top");
  auto *cpu = top->findOrCreateChild("cpu");
  auto *alu = cpu->findOrCreateChild("alu");

  EXPECT_EQ(top->path, "top");
  EXPECT_EQ(cpu->path, "top.cpu");
  EXPECT_EQ(alu->path, "top.cpu.alu");
}

TEST(WaveformScopeTest, FindExistingChild) {
  WaveformScope root("", "");
  auto *child1 = root.findOrCreateChild("child");
  auto *child2 = root.findOrCreateChild("child");

  EXPECT_EQ(child1, child2);
}

//===----------------------------------------------------------------------===//
// VCDFormat Tests
//===----------------------------------------------------------------------===//

TEST(VCDFormatTest, OpenClose) {
  std::string tempFile = "/tmp/test_vcd_openclose.vcd";
  VCDFormat vcd;

  EXPECT_TRUE(vcd.open(tempFile));
  EXPECT_TRUE(vcd.isOpen());

  vcd.close();
  EXPECT_FALSE(vcd.isOpen());

  // Clean up
  std::remove(tempFile.c_str());
}

TEST(VCDFormatTest, WriteHeader) {
  std::string tempFile = "/tmp/test_vcd_header.vcd";
  VCDFormat vcd;

  ASSERT_TRUE(vcd.open(tempFile));
  vcd.writeHeader("2024-01-01", "test", "1ns");
  vcd.beginScope("module", "top");
  vcd.endScope();
  vcd.endDefinitions();
  vcd.close();

  // Verify content
  std::ifstream file(tempFile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  EXPECT_TRUE(content.find("$date") != std::string::npos);
  EXPECT_TRUE(content.find("$version") != std::string::npos);
  EXPECT_TRUE(content.find("$timescale") != std::string::npos);
  EXPECT_TRUE(content.find("$scope module top") != std::string::npos);
  EXPECT_TRUE(content.find("$upscope") != std::string::npos);
  EXPECT_TRUE(content.find("$enddefinitions") != std::string::npos);

  // Clean up
  std::remove(tempFile.c_str());
}

TEST(VCDFormatTest, WriteBitValue) {
  std::string tempFile = "/tmp/test_vcd_bit.vcd";
  VCDFormat vcd;

  ASSERT_TRUE(vcd.open(tempFile));

  TracedSignal sig(1, "clk", "clk", 1);
  sig.vcdId = "!";

  vcd.writeBitValue(sig, true);
  vcd.writeBitValue(sig, false);
  vcd.close();

  // Verify content
  std::ifstream file(tempFile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  EXPECT_TRUE(content.find("1!") != std::string::npos);
  EXPECT_TRUE(content.find("0!") != std::string::npos);

  // Clean up
  std::remove(tempFile.c_str());
}

TEST(VCDFormatTest, WriteVectorValue) {
  std::string tempFile = "/tmp/test_vcd_vector.vcd";
  VCDFormat vcd;

  ASSERT_TRUE(vcd.open(tempFile));

  TracedSignal sig(1, "data", "data", 8);
  sig.vcdId = "\"";

  vcd.writeVectorValue(sig, 0x55); // 01010101
  vcd.close();

  // Verify content
  std::ifstream file(tempFile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  EXPECT_TRUE(content.find("b01010101 \"") != std::string::npos);

  // Clean up
  std::remove(tempFile.c_str());
}

TEST(VCDFormatTest, WriteTime) {
  std::string tempFile = "/tmp/test_vcd_time.vcd";
  VCDFormat vcd;

  ASSERT_TRUE(vcd.open(tempFile));
  vcd.writeTime(0);
  vcd.writeTime(100);
  vcd.writeTime(12345);
  vcd.close();

  // Verify content
  std::ifstream file(tempFile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  EXPECT_TRUE(content.find("#0") != std::string::npos);
  EXPECT_TRUE(content.find("#100") != std::string::npos);
  EXPECT_TRUE(content.find("#12345") != std::string::npos);

  // Clean up
  std::remove(tempFile.c_str());
}

//===----------------------------------------------------------------------===//
// WaveformDumper Tests
//===----------------------------------------------------------------------===//

TEST(WaveformDumperTest, RegisterSignal) {
  WaveformDumperConfig config;
  config.traceAll = true;
  WaveformDumper dumper(config);

  auto handle = dumper.registerSignal("clk", 1);
  EXPECT_NE(handle, 0u);

  auto *sig = dumper.getSignal(handle);
  ASSERT_NE(sig, nullptr);
  EXPECT_EQ(sig->name, "clk");
  EXPECT_EQ(sig->width, 1u);
}

TEST(WaveformDumperTest, ScopeManagement) {
  WaveformDumperConfig config;
  config.traceAll = true;
  WaveformDumper dumper(config);

  dumper.beginScope("top", "module");
  EXPECT_EQ(dumper.getCurrentScopePath(), "top");

  dumper.beginScope("cpu", "module");
  EXPECT_EQ(dumper.getCurrentScopePath(), "top.cpu");

  dumper.endScope();
  EXPECT_EQ(dumper.getCurrentScopePath(), "top");

  dumper.endScope();
  EXPECT_EQ(dumper.getCurrentScopePath(), "");
}

TEST(WaveformDumperTest, TracePatterns) {
  WaveformDumperConfig config;
  config.traceAll = false;
  WaveformDumper dumper(config);

  dumper.addTracePattern("top.cpu.*");
  dumper.addTracePattern("*.clk");

  EXPECT_TRUE(dumper.shouldTrace("top.cpu.reg"));
  EXPECT_TRUE(dumper.shouldTrace("any.path.clk"));
  EXPECT_FALSE(dumper.shouldTrace("top.mem.data"));
}

TEST(WaveformDumperTest, VCDIdentifierGeneration) {
  WaveformDumperConfig config;
  config.traceAll = true;
  WaveformDumper dumper(config);

  auto id1 = dumper.nextVCDIdentifier();
  auto id2 = dumper.nextVCDIdentifier();
  auto id3 = dumper.nextVCDIdentifier();

  // IDs should be unique
  EXPECT_NE(id1, id2);
  EXPECT_NE(id2, id3);
  EXPECT_NE(id1, id3);

  // First ID should be "!"
  EXPECT_EQ(id1, "!");
}

TEST(WaveformDumperTest, ValueUpdates) {
  std::string tempFile = "/tmp/test_waveform_updates.vcd";

  WaveformDumperConfig config;
  config.traceAll = true;
  WaveformDumper dumper(config);

  ASSERT_TRUE(dumper.openVCD(tempFile));

  dumper.beginScope("top", "module");
  auto clkHandle = dumper.registerSignal("clk", 1);
  auto dataHandle = dumper.registerSignal("data", 8);
  dumper.endScope();

  dumper.writeHeader();
  dumper.writeInitialValues();

  dumper.setTime(0);
  dumper.updateBit(clkHandle, false);
  dumper.updateVector(dataHandle, 0x00);

  dumper.setTime(100);
  dumper.updateBit(clkHandle, true);
  dumper.updateVector(dataHandle, 0xAA);

  dumper.setTime(200);
  dumper.updateBit(clkHandle, false);
  dumper.updateVector(dataHandle, 0x55);

  dumper.close();

  // Verify statistics
  const auto &stats = dumper.getStatistics();
  EXPECT_EQ(stats.signalsRegistered, 2u);
  EXPECT_EQ(stats.signalsTraced, 2u);
  EXPECT_GE(stats.valueChanges, 6u);
  EXPECT_GE(stats.timeChanges, 3u);

  // Clean up
  std::remove(tempFile.c_str());
}

TEST(WaveformDumperTest, NoChangeDetection) {
  WaveformDumperConfig config;
  config.traceAll = true;
  WaveformDumper dumper(config);

  std::string tempFile = "/tmp/test_no_change.vcd";
  ASSERT_TRUE(dumper.openVCD(tempFile));

  dumper.beginScope("top", "module");
  auto handle = dumper.registerSignal("data", 8);
  dumper.endScope();

  dumper.writeHeader();
  dumper.writeInitialValues();

  dumper.setTime(0);
  dumper.updateVector(handle, 0x42);

  auto changesBefore = dumper.getStatistics().valueChanges;

  // Update with same value should not record a change
  dumper.updateVector(handle, 0x42);
  dumper.updateVector(handle, 0x42);

  auto changesAfter = dumper.getStatistics().valueChanges;

  // No new changes should be recorded
  EXPECT_EQ(changesBefore, changesAfter);

  dumper.close();
  std::remove(tempFile.c_str());
}

TEST(WaveformDumperTest, SelectiveTracing) {
  WaveformDumperConfig config;
  config.traceAll = false;
  WaveformDumper dumper(config);

  std::string tempFile = "/tmp/test_selective.vcd";
  ASSERT_TRUE(dumper.openVCD(tempFile));

  // Only trace signals matching pattern
  dumper.addTracePattern("*.important*");

  dumper.beginScope("top", "module");
  auto h1 = dumper.registerSignal("important_signal", 8);
  auto h2 = dumper.registerSignal("unimportant_signal", 8);
  dumper.endScope();

  // h1 should be traced, h2 should not
  EXPECT_NE(h1, 0u);
  EXPECT_EQ(h2, 0u); // Returns 0 if not traced

  const auto &stats = dumper.getStatistics();
  EXPECT_EQ(stats.signalsTraced, 1u);

  dumper.close();
  std::remove(tempFile.c_str());
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST(WaveformDumperIntegration, CompleteSimulation) {
  std::string tempFile = "/tmp/test_complete_sim.vcd";

  WaveformDumperConfig config;
  config.traceAll = true;
  config.timescale = "1ps";
  WaveformDumper dumper(config);

  ASSERT_TRUE(dumper.openVCD(tempFile));

  // Build hierarchy
  dumper.beginScope("top", "module");
  auto clk = dumper.registerSignal("clk", 1);
  auto rst = dumper.registerSignal("rst", 1);

  dumper.beginScope("counter", "module");
  auto count = dumper.registerSignal("count", 8, SignalType::Reg);
  auto enable = dumper.registerSignal("enable", 1);
  dumper.endScope();

  dumper.endScope();

  dumper.writeHeader();
  dumper.writeInitialValues();

  // Simulate a simple counter
  dumper.setTime(0);
  dumper.updateBit(clk, false);
  dumper.updateBit(rst, true);
  dumper.updateBit(enable, false);
  dumper.updateVector(count, 0);

  // Release reset
  dumper.setTime(1000);
  dumper.updateBit(rst, false);

  // Enable counter
  dumper.setTime(2000);
  dumper.updateBit(enable, true);

  // Simulate a few clock cycles
  for (int i = 0; i < 5; i++) {
    dumper.setTime(3000 + i * 1000);
    dumper.updateBit(clk, true);
    dumper.updateVector(count, i + 1);

    dumper.setTime(3500 + i * 1000);
    dumper.updateBit(clk, false);
  }

  dumper.close();

  // Verify file was created
  std::ifstream file(tempFile);
  EXPECT_TRUE(file.good());

  // Clean up
  std::remove(tempFile.c_str());
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
