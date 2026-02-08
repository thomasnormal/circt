#include "circt/Support/SMTModel.h"
#include "gtest/gtest.h"

extern "C" void circt_smt_print_model_header();
extern "C" void circt_smt_print_model_value(const char *name,
                                            const char *value);

using namespace circt;

TEST(SMTModelTest, CapturesAndResetsModelValues) {
  resetCapturedSMTModelValues();
  EXPECT_TRUE(getCapturedSMTModelValues().empty());

  circt_smt_print_model_header();
  circt_smt_print_model_value("sig", "#b1");
  circt_smt_print_model_value("wide", "(_ bv10 8)");

  auto model = getCapturedSMTModelValues();
  auto sigIt = model.find("sig");
  ASSERT_NE(sigIt, model.end());
  EXPECT_EQ(sigIt->second, "1'd1");

  auto wideIt = model.find("wide");
  ASSERT_NE(wideIt, model.end());
  EXPECT_EQ(wideIt->second, "8'h0A");

  circt_smt_print_model_value("sig", "#b0");
  model = getCapturedSMTModelValues();
  sigIt = model.find("sig");
  ASSERT_NE(sigIt, model.end());
  EXPECT_EQ(sigIt->second, "1'd0");

  resetCapturedSMTModelValues();
  EXPECT_TRUE(getCapturedSMTModelValues().empty());
}
