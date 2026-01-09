//===- TLMPortsTest.cpp - Tests for TLM port infrastructure ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/TLMPorts.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Test Transaction Class
//===----------------------------------------------------------------------===//

class TestTransaction : public TLMTransaction {
public:
  TestTransaction() = default;
  explicit TestTransaction(int data) : data(data) {}

  std::unique_ptr<TLMTransaction> clone() const override {
    auto copy = std::make_unique<TestTransaction>(data);
    copy->setId(getId());
    copy->setTimestamp(getTimestamp());
    return copy;
  }

  int data = 0;
};

//===----------------------------------------------------------------------===//
// TLMFifo Tests
//===----------------------------------------------------------------------===//

TEST(TLMFifoTest, DefaultConstruction) {
  TLMFifo<int> fifo;

  EXPECT_TRUE(fifo.isEmpty());
  EXPECT_EQ(fifo.size(), 0u);
  EXPECT_FALSE(fifo.isFull());
  EXPECT_TRUE(fifo.canPut());
}

TEST(TLMFifoTest, BoundedFifo) {
  TLMFifo<int> fifo(3);

  EXPECT_EQ(fifo.getMaxDepth(), 3u);
  EXPECT_EQ(fifo.freeSlots(), 3u);

  fifo.tryPut(1);
  fifo.tryPut(2);
  fifo.tryPut(3);

  EXPECT_TRUE(fifo.isFull());
  EXPECT_FALSE(fifo.canPut());
  EXPECT_FALSE(fifo.tryPut(4)); // Should fail
}

TEST(TLMFifoTest, UnlimitedFifo) {
  TLMFifo<int> fifo(0); // 0 = unlimited

  EXPECT_EQ(fifo.getMaxDepth(), 0u);
  EXPECT_FALSE(fifo.isFull());

  // Can put many items
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(fifo.tryPut(i));
  }

  EXPECT_EQ(fifo.size(), 100u);
  EXPECT_FALSE(fifo.isFull());
}

TEST(TLMFifoTest, PutAndGet) {
  TLMFifo<int> fifo;

  EXPECT_TRUE(fifo.tryPut(42));
  EXPECT_TRUE(fifo.tryPut(43));

  EXPECT_EQ(fifo.size(), 2u);
  EXPECT_TRUE(fifo.canGet());

  int val;
  EXPECT_TRUE(fifo.tryGet(val));
  EXPECT_EQ(val, 42);

  EXPECT_TRUE(fifo.tryGet(val));
  EXPECT_EQ(val, 43);

  EXPECT_FALSE(fifo.tryGet(val)); // Empty
}

TEST(TLMFifoTest, Peek) {
  TLMFifo<int> fifo;

  fifo.tryPut(100);

  int val;
  EXPECT_TRUE(fifo.tryPeek(val));
  EXPECT_EQ(val, 100);

  // Peek shouldn't remove
  EXPECT_EQ(fifo.size(), 1u);

  // Get should still return same value
  EXPECT_TRUE(fifo.tryGet(val));
  EXPECT_EQ(val, 100);
}

TEST(TLMFifoTest, Flush) {
  TLMFifo<int> fifo;

  fifo.tryPut(1);
  fifo.tryPut(2);
  fifo.tryPut(3);

  EXPECT_EQ(fifo.size(), 3u);

  fifo.flush();

  EXPECT_TRUE(fifo.isEmpty());
  EXPECT_EQ(fifo.size(), 0u);
}

TEST(TLMFifoTest, FifoOrder) {
  TLMFifo<int> fifo;

  for (int i = 0; i < 5; ++i) {
    fifo.tryPut(i);
  }

  for (int i = 0; i < 5; ++i) {
    int val;
    EXPECT_TRUE(fifo.tryGet(val));
    EXPECT_EQ(val, i);
  }
}

TEST(TLMFifoTest, Callbacks) {
  TLMFifo<int> fifo;

  bool writtenCalled = false;
  bool readCalled = false;

  fifo.setDataWrittenCallback([&]() { writtenCalled = true; });
  fifo.setDataReadCallback([&]() { readCalled = true; });

  fifo.tryPut(1);
  EXPECT_TRUE(writtenCalled);

  int val;
  fifo.tryGet(val);
  EXPECT_TRUE(readCalled);
}

TEST(TLMFifoTest, ComplexType) {
  TLMFifo<TestTransaction> fifo;

  TestTransaction tx1(100);
  TestTransaction tx2(200);

  EXPECT_TRUE(fifo.tryPut(tx1));
  EXPECT_TRUE(fifo.tryPut(tx2));

  TestTransaction result;
  EXPECT_TRUE(fifo.tryGet(result));
  EXPECT_EQ(result.data, 100);
}

//===----------------------------------------------------------------------===//
// TLMAnalysisPort Tests
//===----------------------------------------------------------------------===//

TEST(TLMAnalysisPortTest, Construction) {
  TLMAnalysisPort<int> port("test_port");

  EXPECT_EQ(port.getName(), "test_port");
  EXPECT_EQ(port.getSubscriberCount(), 0u);
}

TEST(TLMAnalysisPortTest, Connect) {
  TLMAnalysisPort<int> port;
  TLMAnalysisExport<int> export1;
  TLMAnalysisExport<int> export2;

  port.connect(&export1);
  port.connect(&export2);

  EXPECT_EQ(port.getSubscriberCount(), 2u);
}

TEST(TLMAnalysisPortTest, Disconnect) {
  TLMAnalysisPort<int> port;
  TLMAnalysisExport<int> export1;

  port.connect(&export1);
  EXPECT_EQ(port.getSubscriberCount(), 1u);

  port.disconnect(&export1);
  EXPECT_EQ(port.getSubscriberCount(), 0u);
}

TEST(TLMAnalysisPortTest, Broadcast) {
  TLMAnalysisPort<int> port;

  std::vector<int> received1;
  std::vector<int> received2;

  TLMAnalysisExport<int> export1([&](const int& val) { received1.push_back(val); });
  TLMAnalysisExport<int> export2([&](const int& val) { received2.push_back(val); });

  port.connect(&export1);
  port.connect(&export2);

  port.write(100);
  port.write(200);

  EXPECT_EQ(received1.size(), 2u);
  EXPECT_EQ(received2.size(), 2u);

  EXPECT_EQ(received1[0], 100);
  EXPECT_EQ(received1[1], 200);
  EXPECT_EQ(received2[0], 100);
  EXPECT_EQ(received2[1], 200);
}

TEST(TLMAnalysisPortTest, WriteCount) {
  TLMAnalysisPort<int> port;
  TLMAnalysisExport<int> export1;

  port.connect(&export1);

  port.write(1);
  port.write(2);
  port.write(3);

  EXPECT_EQ(port.getWriteCount(), 3u);
}

//===----------------------------------------------------------------------===//
// TLMAnalysisExport Tests
//===----------------------------------------------------------------------===//

TEST(TLMAnalysisExportTest, DefaultConstruction) {
  TLMAnalysisExport<int> exp;

  // Should not crash when written to without callback
  exp.write(42);

  EXPECT_EQ(exp.getReceiveCount(), 1u);
}

TEST(TLMAnalysisExportTest, WithCallback) {
  int lastValue = 0;

  TLMAnalysisExport<int> exp([&](const int& val) {
    lastValue = val;
  });

  exp.write(42);
  EXPECT_EQ(lastValue, 42);

  exp.write(100);
  EXPECT_EQ(lastValue, 100);
}

TEST(TLMAnalysisExportTest, ReceiveCount) {
  TLMAnalysisExport<int> exp;

  exp.write(1);
  exp.write(2);
  exp.write(3);

  EXPECT_EQ(exp.getReceiveCount(), 3u);
}

//===----------------------------------------------------------------------===//
// TLMPort Tests
//===----------------------------------------------------------------------===//

TEST(TLMPortTest, Connection) {
  TLMPutIf<int>* interface = nullptr;
  TLMPort<TLMPutIf<int>> port("test_port");

  EXPECT_FALSE(port.isConnected());
  EXPECT_EQ(port.getInterface(), nullptr);

  TLMFifo<int> fifo;
  port.connect(&fifo);

  EXPECT_TRUE(port.isConnected());
  EXPECT_EQ(port.getInterface(), &fifo);
}

TEST(TLMPortTest, ArrowOperator) {
  TLMFifo<int> fifo;
  TLMPort<TLMPutIf<int>> port;

  port.connect(&fifo);

  // Use arrow operator to access interface methods
  EXPECT_TRUE(port->canPut());
  port->tryPut(42);

  EXPECT_EQ(fifo.size(), 1u);
}

//===----------------------------------------------------------------------===//
// TLMSeqItemPullPort Tests
//===----------------------------------------------------------------------===//

TEST(TLMSeqItemPullPortTest, Construction) {
  TLMSeqItemPullPort<int> port("seq_item_port");

  EXPECT_EQ(port.getName(), "seq_item_port");
}

TEST(TLMSeqItemPullPortTest, GetNextItem) {
  TLMSeqItemPullPort<int> port;

  std::queue<int> items;
  items.push(1);
  items.push(2);
  items.push(3);

  port.setGetNextItemCallback([&](int& item) -> bool {
    if (items.empty())
      return false;
    item = items.front();
    items.pop();
    return true;
  });

  int item;
  EXPECT_TRUE(port.getNextItem(item));
  EXPECT_EQ(item, 1);

  EXPECT_TRUE(port.getNextItem(item));
  EXPECT_EQ(item, 2);

  EXPECT_TRUE(port.getNextItem(item));
  EXPECT_EQ(item, 3);

  EXPECT_FALSE(port.getNextItem(item));
}

TEST(TLMSeqItemPullPortTest, ItemDone) {
  TLMSeqItemPullPort<int> port;

  std::vector<int> completedItems;

  port.setItemDoneCallback([&](const int& item) {
    completedItems.push_back(item);
  });

  port.itemDone(10);
  port.itemDone(20);

  EXPECT_EQ(completedItems.size(), 2u);
  EXPECT_EQ(completedItems[0], 10);
  EXPECT_EQ(completedItems[1], 20);
}

//===----------------------------------------------------------------------===//
// TLMTransaction Tests
//===----------------------------------------------------------------------===//

TEST(TLMTransactionTest, DefaultConstruction) {
  TLMTransaction tx;

  EXPECT_EQ(tx.getId(), 0u);
  EXPECT_EQ(tx.getTimestamp(), 0u);
}

TEST(TLMTransactionTest, SetId) {
  TLMTransaction tx;

  tx.setId(12345);
  EXPECT_EQ(tx.getId(), 12345u);
}

TEST(TLMTransactionTest, SetTimestamp) {
  TLMTransaction tx;

  tx.setTimestamp(1000000);
  EXPECT_EQ(tx.getTimestamp(), 1000000u);
}

TEST(TLMTransactionTest, Clone) {
  TestTransaction tx(42);
  tx.setId(100);
  tx.setTimestamp(5000);

  auto clone = tx.clone();

  EXPECT_EQ(clone->getId(), 100u);
  EXPECT_EQ(clone->getTimestamp(), 5000u);

  // Check derived data was preserved
  TestTransaction* derived = static_cast<TestTransaction*>(clone.get());
  EXPECT_EQ(derived->data, 42);
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST(TLMIntegrationTest, ProducerConsumer) {
  TLMFifo<int> fifo(10);
  TLMPort<TLMPutIf<int>> producerPort;
  TLMPort<TLMGetIf<int>> consumerPort;

  producerPort.connect(&fifo);
  consumerPort.connect(&fifo);

  // Producer writes
  for (int i = 0; i < 5; ++i) {
    producerPort->tryPut(i * 10);
  }

  EXPECT_EQ(fifo.size(), 5u);

  // Consumer reads
  for (int i = 0; i < 5; ++i) {
    int val;
    EXPECT_TRUE(consumerPort->tryGet(val));
    EXPECT_EQ(val, i * 10);
  }
}

TEST(TLMIntegrationTest, AnalysisChain) {
  TLMAnalysisPort<int> source;
  TLMAnalysisPort<int> intermediate;
  std::vector<int> finalResults;

  TLMAnalysisExport<int> intermediateExport([&](const int& val) {
    intermediate.write(val * 2);
  });

  TLMAnalysisExport<int> finalExport([&](const int& val) {
    finalResults.push_back(val);
  });

  source.connect(&intermediateExport);
  intermediate.connect(&finalExport);

  source.write(5);
  source.write(10);

  EXPECT_EQ(finalResults.size(), 2u);
  EXPECT_EQ(finalResults[0], 10); // 5 * 2
  EXPECT_EQ(finalResults[1], 20); // 10 * 2
}
