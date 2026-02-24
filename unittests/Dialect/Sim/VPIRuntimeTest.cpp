//===- VPIRuntimeTest.cpp - Unit tests for VPIRuntime ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/VPIRuntime.h"
#include "gtest/gtest.h"

using namespace circt::sim;

extern "C" vpiHandle vpi_register_cb(p_cb_data cb_data_p);

namespace {

static PLI_INT32 noopCallback(t_cb_data *cbData) {
  (void)cbData;
  return 0;
}

class VPIRuntimeRegisterCbTest : public ::testing::Test {
protected:
  VPIRuntime &vpi = VPIRuntime::getInstance();
  bool oldActive = false;

  void SetUp() override {
    oldActive = vpi.isActive();
    vpi.setActive(false);
  }

  void TearDown() override { vpi.setActive(oldActive); }
};

TEST_F(VPIRuntimeRegisterCbTest,
       AllowsCbStartOfSimulationRegistrationWhileInactive) {
  t_cb_data startCb = {};
  startCb.reason = cbStartOfSimulation;
  startCb.cb_rtn = noopCallback;

  vpiHandle handle = vpi_register_cb(&startCb);
  ASSERT_NE(handle, nullptr);

  uint32_t cbId = VPIRuntime::getHandleId(handle);
  EXPECT_EQ(vpi.removeCb(cbId), 1);
}

TEST_F(VPIRuntimeRegisterCbTest, RejectsNonStartCallbacksWhileInactive) {
  t_cb_data endCb = {};
  endCb.reason = cbEndOfSimulation;
  endCb.cb_rtn = noopCallback;

  EXPECT_EQ(vpi_register_cb(&endCb), nullptr);
}

TEST_F(VPIRuntimeRegisterCbTest, RejectsNullCallbackDataWhileInactive) {
  EXPECT_EQ(vpi_register_cb(nullptr), nullptr);
}

} // namespace
