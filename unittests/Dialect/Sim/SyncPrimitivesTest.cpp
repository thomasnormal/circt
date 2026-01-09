//===- SyncPrimitivesTest.cpp - Semaphore/Mailbox unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "gtest/gtest.h"

using namespace circt::sim;

namespace {

class SyncPrimitivesTest : public ::testing::Test {
protected:
  void SetUp() override {
    scheduler = std::make_unique<ProcessScheduler>();
    syncManager = std::make_unique<SyncPrimitivesManager>(*scheduler);
  }

  std::unique_ptr<ProcessScheduler> scheduler;
  std::unique_ptr<SyncPrimitivesManager> syncManager;
};

//===----------------------------------------------------------------------===//
// Semaphore Tests
//===----------------------------------------------------------------------===//

TEST_F(SyncPrimitivesTest, SemaphoreCreate) {
  SemaphoreId sem = syncManager->createSemaphore(5);
  ASSERT_NE(sem, InvalidSemaphoreId);

  Semaphore *semaphore = syncManager->getSemaphore(sem);
  ASSERT_NE(semaphore, nullptr);
  EXPECT_EQ(semaphore->getKeyCount(), 5);
}

TEST_F(SyncPrimitivesTest, SemaphoreTryGet) {
  SemaphoreId sem = syncManager->createSemaphore(3);

  // Should succeed with keys available
  EXPECT_TRUE(syncManager->semaphoreTryGet(sem, 2));

  Semaphore *semaphore = syncManager->getSemaphore(sem);
  EXPECT_EQ(semaphore->getKeyCount(), 1);

  // Should fail with insufficient keys
  EXPECT_FALSE(syncManager->semaphoreTryGet(sem, 2));
  EXPECT_EQ(semaphore->getKeyCount(), 1);

  // Should succeed with exactly available keys
  EXPECT_TRUE(syncManager->semaphoreTryGet(sem, 1));
  EXPECT_EQ(semaphore->getKeyCount(), 0);
}

TEST_F(SyncPrimitivesTest, SemaphorePut) {
  SemaphoreId sem = syncManager->createSemaphore(0);

  syncManager->semaphorePut(sem, 5);

  Semaphore *semaphore = syncManager->getSemaphore(sem);
  EXPECT_EQ(semaphore->getKeyCount(), 5);
}

TEST_F(SyncPrimitivesTest, SemaphoreBlocking) {
  SemaphoreId sem = syncManager->createSemaphore(0);
  ProcessId proc = scheduler->registerProcess("waiter", []() {});
  scheduler->getProcess(proc)->setState(ProcessState::Running);

  // Get should block (process waiting)
  syncManager->semaphoreGet(sem, proc, 1);
  EXPECT_EQ(scheduler->getProcess(proc)->getState(), ProcessState::Waiting);

  // Put should resume the waiter
  syncManager->semaphorePut(sem, 1);
  EXPECT_EQ(scheduler->getProcess(proc)->getState(), ProcessState::Ready);
}

TEST_F(SyncPrimitivesTest, SemaphoreBinaryMutex) {
  // Create a binary semaphore (mutex)
  SemaphoreId mutex = syncManager->createSemaphore(1);

  // First acquire should succeed
  EXPECT_TRUE(syncManager->semaphoreTryGet(mutex, 1));

  // Second acquire should fail (mutex held)
  EXPECT_FALSE(syncManager->semaphoreTryGet(mutex, 1));

  // Release
  syncManager->semaphorePut(mutex, 1);

  // Now acquire should succeed again
  EXPECT_TRUE(syncManager->semaphoreTryGet(mutex, 1));
}

//===----------------------------------------------------------------------===//
// Mailbox Tests
//===----------------------------------------------------------------------===//

TEST_F(SyncPrimitivesTest, MailboxCreate) {
  MailboxId mbox = syncManager->createMailbox();
  ASSERT_NE(mbox, InvalidMailboxId);

  Mailbox *mailbox = syncManager->getMailbox(mbox);
  ASSERT_NE(mailbox, nullptr);
  EXPECT_FALSE(mailbox->isBounded());
  EXPECT_TRUE(mailbox->isEmpty());
}

TEST_F(SyncPrimitivesTest, MailboxBoundedCreate) {
  MailboxId mbox = syncManager->createMailbox(10);

  Mailbox *mailbox = syncManager->getMailbox(mbox);
  ASSERT_NE(mailbox, nullptr);
  EXPECT_TRUE(mailbox->isBounded());
}

TEST_F(SyncPrimitivesTest, MailboxTryPutGet) {
  MailboxId mbox = syncManager->createMailbox();

  // Put a message
  EXPECT_TRUE(syncManager->mailboxTryPut(mbox, 42));

  // Get the message
  uint64_t msg;
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_EQ(msg, 42u);

  // Mailbox should be empty now
  EXPECT_FALSE(syncManager->mailboxTryGet(mbox, msg));
}

TEST_F(SyncPrimitivesTest, MailboxPeek) {
  MailboxId mbox = syncManager->createMailbox();

  syncManager->mailboxTryPut(mbox, 123);

  uint64_t msg;
  // Peek should return the message without removing
  EXPECT_TRUE(syncManager->mailboxPeek(mbox, msg));
  EXPECT_EQ(msg, 123u);

  // Message should still be there
  EXPECT_TRUE(syncManager->mailboxPeek(mbox, msg));
  EXPECT_EQ(msg, 123u);

  // Get should remove it
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_EQ(msg, 123u);

  // Now peek should fail
  EXPECT_FALSE(syncManager->mailboxPeek(mbox, msg));
}

TEST_F(SyncPrimitivesTest, MailboxNum) {
  MailboxId mbox = syncManager->createMailbox();

  EXPECT_EQ(syncManager->mailboxNum(mbox), 0u);

  syncManager->mailboxTryPut(mbox, 1);
  EXPECT_EQ(syncManager->mailboxNum(mbox), 1u);

  syncManager->mailboxTryPut(mbox, 2);
  syncManager->mailboxTryPut(mbox, 3);
  EXPECT_EQ(syncManager->mailboxNum(mbox), 3u);

  uint64_t msg;
  syncManager->mailboxTryGet(mbox, msg);
  EXPECT_EQ(syncManager->mailboxNum(mbox), 2u);
}

TEST_F(SyncPrimitivesTest, MailboxFIFO) {
  MailboxId mbox = syncManager->createMailbox();

  syncManager->mailboxTryPut(mbox, 1);
  syncManager->mailboxTryPut(mbox, 2);
  syncManager->mailboxTryPut(mbox, 3);

  uint64_t msg;
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_EQ(msg, 1u);
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_EQ(msg, 2u);
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_EQ(msg, 3u);
}

TEST_F(SyncPrimitivesTest, MailboxBounded) {
  MailboxId mbox = syncManager->createMailbox(2);

  Mailbox *mailbox = syncManager->getMailbox(mbox);

  EXPECT_TRUE(syncManager->mailboxTryPut(mbox, 1));
  EXPECT_FALSE(mailbox->isFull());

  EXPECT_TRUE(syncManager->mailboxTryPut(mbox, 2));
  EXPECT_TRUE(mailbox->isFull());

  // Should fail when full
  EXPECT_FALSE(syncManager->mailboxTryPut(mbox, 3));

  // After getting one, should be able to put again
  uint64_t msg;
  EXPECT_TRUE(syncManager->mailboxTryGet(mbox, msg));
  EXPECT_TRUE(syncManager->mailboxTryPut(mbox, 3));
}

//===----------------------------------------------------------------------===//
// Semaphore Direct API Tests
//===----------------------------------------------------------------------===//

TEST_F(SyncPrimitivesTest, SemaphoreDirectAPI) {
  Semaphore sem(1, 3);

  EXPECT_EQ(sem.getId(), 1u);
  EXPECT_EQ(sem.getKeyCount(), 3);
  EXPECT_FALSE(sem.hasWaiters());

  // Try get
  EXPECT_TRUE(sem.tryGet(2));
  EXPECT_EQ(sem.getKeyCount(), 1);
  EXPECT_FALSE(sem.tryGet(2));

  // Put
  sem.put(4);
  EXPECT_EQ(sem.getKeyCount(), 5);
}

TEST_F(SyncPrimitivesTest, SemaphoreWaitQueue) {
  Semaphore sem(1, 0);

  sem.addWaiter(100, 1);
  sem.addWaiter(200, 2);
  EXPECT_TRUE(sem.hasWaiters());

  // Not enough keys to satisfy
  EXPECT_EQ(sem.trySatisfyNextWaiter(), InvalidProcessId);

  // Add a key
  sem.put(1);
  EXPECT_EQ(sem.trySatisfyNextWaiter(), 100u);

  // Add more keys
  sem.put(2);
  EXPECT_EQ(sem.trySatisfyNextWaiter(), 200u);

  EXPECT_FALSE(sem.hasWaiters());
}

//===----------------------------------------------------------------------===//
// Mailbox Direct API Tests
//===----------------------------------------------------------------------===//

TEST_F(SyncPrimitivesTest, MailboxDirectAPI) {
  Mailbox mbox(1, 0); // unbounded

  EXPECT_EQ(mbox.getId(), 1u);
  EXPECT_FALSE(mbox.isBounded());
  EXPECT_FALSE(mbox.isFull());
  EXPECT_TRUE(mbox.isEmpty());

  mbox.put(42);
  EXPECT_FALSE(mbox.isEmpty());
  EXPECT_EQ(mbox.getMessageCount(), 1u);

  uint64_t msg;
  EXPECT_TRUE(mbox.tryGet(msg));
  EXPECT_EQ(msg, 42u);
  EXPECT_TRUE(mbox.isEmpty());
}

TEST_F(SyncPrimitivesTest, MailboxBoundedDirectAPI) {
  Mailbox mbox(1, 2); // bound = 2

  EXPECT_TRUE(mbox.isBounded());

  EXPECT_TRUE(mbox.tryPut(1));
  EXPECT_FALSE(mbox.isFull());
  EXPECT_TRUE(mbox.tryPut(2));
  EXPECT_TRUE(mbox.isFull());
  EXPECT_FALSE(mbox.tryPut(3)); // Should fail
}

TEST_F(SyncPrimitivesTest, MailboxWaitQueues) {
  Mailbox mbox(1, 1);

  // Add a get waiter (mailbox empty)
  mbox.addGetWaiter(100);

  // Put a message - should satisfy waiter
  mbox.put(42);
  uint64_t msg;
  EXPECT_EQ(mbox.trySatisfyGetWaiter(msg), 100u);
  EXPECT_EQ(msg, 42u);

  // Fill the mailbox
  mbox.put(1);

  // Add a put waiter (mailbox full)
  mbox.addPutWaiter(200, 2);

  // Get a message - should satisfy put waiter
  EXPECT_TRUE(mbox.tryGet(msg));
  EXPECT_EQ(mbox.trySatisfyPutWaiter(), 200u);

  // The put waiter's message should now be in the mailbox
  EXPECT_TRUE(mbox.tryGet(msg));
  EXPECT_EQ(msg, 2u);
}

} // namespace
