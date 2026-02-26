//===- LowerTaggedIndirectCallsTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerTaggedIndirectCalls.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(LowerTaggedIndirectCallsTest, InvokeUnwindPhiUsesInvokeBlocks) {
  constexpr StringLiteral ir = R"IR(
declare i32 @__gxx_personality_v0(...)

@__circt_sim_func_entries = global [1 x ptr] [ptr null]

define i32 @test(ptr %fp, i32 %x) personality ptr @__gxx_personality_v0 {
entry:
  %tag = add i32 %x, 1
  %res = invoke i32 %fp(i32 %x) to label %normal unwind label %lpad

normal:
  ret i32 %res

lpad:
  %u = phi i32 [ %tag, %entry ]
  %lp = landingpad { ptr, i32 } cleanup
  %sink = add i32 %u, 0
  resume { ptr, i32 } %lp
}
)IR";

  LLVMContext ctx;
  SMDiagnostic parseErr;
  std::unique_ptr<Module> m = parseAssemblyString(ir, parseErr, ctx);
  ASSERT_TRUE(m) << "failed to parse test IR";

  std::string verifyBefore;
  raw_string_ostream verifyBeforeOS(verifyBefore);
  ASSERT_FALSE(verifyModule(*m, &verifyBeforeOS))
      << "input IR must be valid before lowering:\n"
      << verifyBeforeOS.str();

  runLowerTaggedIndirectCalls(*m);

  std::string verifyAfter;
  raw_string_ostream verifyAfterOS(verifyAfter);
  EXPECT_FALSE(verifyModule(*m, &verifyAfterOS))
      << "lowered IR should remain valid:\n"
      << verifyAfterOS.str();

  auto *f = m->getFunction("test");
  ASSERT_NE(f, nullptr);

  BasicBlock *lpad = nullptr;
  for (auto &bb : *f)
    if (bb.getName() == "lpad") {
      lpad = &bb;
      break;
    }
  ASSERT_NE(lpad, nullptr);

  auto *phi = dyn_cast<PHINode>(lpad->begin());
  ASSERT_NE(phi, nullptr) << "expected landing-pad PHI to remain after rewrite";
  EXPECT_EQ(phi->getNumIncomingValues(), 2u);

  bool sawTaggedInvoke = false;
  bool sawDirectCall = false;
  bool sawTaggedCall = false;
  for (unsigned i = 0, e = phi->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *incoming = phi->getIncomingBlock(i);
    ASSERT_NE(incoming, nullptr);
    StringRef name = incoming->getName();
    if (name == "tagged_invoke")
      sawTaggedInvoke = true;
    if (name == "direct_call")
      sawDirectCall = true;
    if (name == "tagged_call")
      sawTaggedCall = true;
  }

  EXPECT_TRUE(sawTaggedInvoke);
  EXPECT_TRUE(sawDirectCall);
  EXPECT_FALSE(sawTaggedCall);
}

} // namespace
