//===- UVMFactoryTest.cpp - Tests for UVMFactory --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/UVMFactory.h"
#include "gtest/gtest.h"
#include <string>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Test Classes for Factory
//===----------------------------------------------------------------------===//

class BaseComponent {
public:
  virtual ~BaseComponent() = default;
  virtual std::string getTypeName() const { return "BaseComponent"; }
  int value = 0;
};

class DerivedComponent : public BaseComponent {
public:
  std::string getTypeName() const override { return "DerivedComponent"; }
};

class AnotherComponent : public BaseComponent {
public:
  std::string getTypeName() const override { return "AnotherComponent"; }
};

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class UVMFactoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    factory = std::make_unique<UVMFactory>();
  }

  void TearDown() override {
    factory.reset();
  }

  std::unique_ptr<UVMFactory> factory;
};

//===----------------------------------------------------------------------===//
// Type Registration Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, RegisterType) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });

  EXPECT_TRUE(factory->isTypeRegistered("BaseComponent"));
  EXPECT_FALSE(factory->isTypeRegistered("NonexistentType"));
}

TEST_F(UVMFactoryTest, RegisterTypeWithDestructor) {
  bool destructorCalled = false;

  factory->registerType(
      "TrackedComponent",
      []() -> void* { return new BaseComponent(); },
      [&](void* p) {
        destructorCalled = true;
        delete static_cast<BaseComponent*>(p);
      });

  void* obj = factory->createByName("TrackedComponent");
  ASSERT_NE(obj, nullptr);

  factory->destroy("TrackedComponent", obj);
  EXPECT_TRUE(destructorCalled);
}

TEST_F(UVMFactoryTest, GetTypeInfo) {
  factory->registerType("TestType", []() -> void* {
    return new BaseComponent();
  });

  const UVMTypeInfo* info = factory->getTypeInfo("TestType");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->typeName, "TestType");

  const UVMTypeInfo* notFound = factory->getTypeInfo("NotFound");
  EXPECT_EQ(notFound, nullptr);
}

//===----------------------------------------------------------------------===//
// Instance Creation Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, CreateByName) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });

  void* obj = factory->createByName("BaseComponent");
  ASSERT_NE(obj, nullptr);

  BaseComponent* comp = static_cast<BaseComponent*>(obj);
  EXPECT_EQ(comp->getTypeName(), "BaseComponent");

  delete comp;
}

TEST_F(UVMFactoryTest, CreateUnregisteredType) {
  void* obj = factory->createByName("UnregisteredType");
  EXPECT_EQ(obj, nullptr);
}

TEST_F(UVMFactoryTest, CreateByNameWithPath) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });

  void* obj = factory->createByName("BaseComponent", "top.env.agent");
  ASSERT_NE(obj, nullptr);

  delete static_cast<BaseComponent*>(obj);
}

TEST_F(UVMFactoryTest, CreateObject) {
  factory->registerType("Driver", []() -> void* {
    return new BaseComponent();
  });

  void* obj = factory->createObject("Driver", "top.env", "driver0");
  ASSERT_NE(obj, nullptr);

  delete static_cast<BaseComponent*>(obj);
}

//===----------------------------------------------------------------------===//
// Type Override Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, SetTypeOverride) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });

  factory->setTypeOverride("BaseComponent", "DerivedComponent");

  void* obj = factory->createByName("BaseComponent");
  ASSERT_NE(obj, nullptr);

  BaseComponent* comp = static_cast<BaseComponent*>(obj);
  EXPECT_EQ(comp->getTypeName(), "DerivedComponent");

  delete comp;
}

TEST_F(UVMFactoryTest, GetTypeOverride) {
  factory->setTypeOverride("OriginalType", "OverrideType");

  EXPECT_EQ(factory->getTypeOverride("OriginalType"), "OverrideType");
  EXPECT_EQ(factory->getTypeOverride("NoOverride"), "NoOverride");
}

TEST_F(UVMFactoryTest, RemoveTypeOverride) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });

  factory->setTypeOverride("BaseComponent", "DerivedComponent");
  factory->removeTypeOverride("BaseComponent");

  void* obj = factory->createByName("BaseComponent");
  ASSERT_NE(obj, nullptr);

  BaseComponent* comp = static_cast<BaseComponent*>(obj);
  EXPECT_EQ(comp->getTypeName(), "BaseComponent");

  delete comp;
}

TEST_F(UVMFactoryTest, TypeOverrideChain) {
  factory->registerType("A", []() -> void* { return new BaseComponent(); });
  factory->registerType("B", []() -> void* { return new DerivedComponent(); });
  factory->registerType("C", []() -> void* { return new AnotherComponent(); });

  // A -> B -> C
  factory->setTypeOverride("A", "B");
  factory->setTypeOverride("B", "C");

  void* obj = factory->createByName("A");
  ASSERT_NE(obj, nullptr);

  BaseComponent* comp = static_cast<BaseComponent*>(obj);
  EXPECT_EQ(comp->getTypeName(), "AnotherComponent");

  delete comp;
}

TEST_F(UVMFactoryTest, TypeOverrideNoReplace) {
  factory->setTypeOverride("Type1", "Override1");
  factory->setTypeOverride("Type1", "Override2", false); // Don't replace

  EXPECT_EQ(factory->getTypeOverride("Type1"), "Override1");
}

//===----------------------------------------------------------------------===//
// Instance Override Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, SetInstOverride) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });

  factory->setInstOverride("top.env.agent.driver", "BaseComponent",
                           "DerivedComponent");

  void* obj = factory->createByName("BaseComponent", "top.env.agent.driver");
  ASSERT_NE(obj, nullptr);

  BaseComponent* comp = static_cast<BaseComponent*>(obj);
  EXPECT_EQ(comp->getTypeName(), "DerivedComponent");

  delete comp;
}

TEST_F(UVMFactoryTest, InstOverrideWithWildcard) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });

  factory->setInstOverride("top.env.*.driver", "BaseComponent",
                           "DerivedComponent");

  // Should match any agent
  void* obj1 = factory->createByName("BaseComponent", "top.env.agent0.driver");
  ASSERT_NE(obj1, nullptr);
  EXPECT_EQ(static_cast<BaseComponent*>(obj1)->getTypeName(), "DerivedComponent");

  void* obj2 = factory->createByName("BaseComponent", "top.env.agent1.driver");
  ASSERT_NE(obj2, nullptr);
  EXPECT_EQ(static_cast<BaseComponent*>(obj2)->getTypeName(), "DerivedComponent");

  delete static_cast<BaseComponent*>(obj1);
  delete static_cast<BaseComponent*>(obj2);
}

TEST_F(UVMFactoryTest, InstOverridePriority) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });
  factory->registerType("AnotherComponent", []() -> void* {
    return new AnotherComponent();
  });

  // Type override
  factory->setTypeOverride("BaseComponent", "DerivedComponent");

  // Instance override (should take priority)
  factory->setInstOverride("top.special", "BaseComponent", "AnotherComponent");

  // Regular path uses type override
  void* obj1 = factory->createByName("BaseComponent", "top.normal");
  EXPECT_EQ(static_cast<BaseComponent*>(obj1)->getTypeName(), "DerivedComponent");

  // Special path uses instance override
  void* obj2 = factory->createByName("BaseComponent", "top.special");
  EXPECT_EQ(static_cast<BaseComponent*>(obj2)->getTypeName(), "AnotherComponent");

  delete static_cast<BaseComponent*>(obj1);
  delete static_cast<BaseComponent*>(obj2);
}

TEST_F(UVMFactoryTest, RemoveInstOverride) {
  factory->registerType("BaseComponent", []() -> void* {
    return new BaseComponent();
  });
  factory->registerType("DerivedComponent", []() -> void* {
    return new DerivedComponent();
  });

  factory->setInstOverride("top.path", "BaseComponent", "DerivedComponent");
  factory->removeInstOverride("top.path", "BaseComponent");

  void* obj = factory->createByName("BaseComponent", "top.path");
  EXPECT_EQ(static_cast<BaseComponent*>(obj)->getTypeName(), "BaseComponent");

  delete static_cast<BaseComponent*>(obj);
}

//===----------------------------------------------------------------------===//
// Override Lookup Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, FindOverrideWithInstAndType) {
  factory->registerType("A", []() -> void* { return new BaseComponent(); });
  factory->registerType("B", []() -> void* { return new DerivedComponent(); });
  factory->registerType("C", []() -> void* { return new AnotherComponent(); });

  factory->setTypeOverride("A", "B");
  factory->setInstOverride("special.path", "A", "C");

  // Without path, should use type override
  EXPECT_EQ(factory->findOverride("A", ""), "B");

  // With non-matching path, should use type override
  EXPECT_EQ(factory->findOverride("A", "other.path"), "B");

  // With matching path, should use instance override
  EXPECT_EQ(factory->findOverride("A", "special.path"), "C");
}

//===----------------------------------------------------------------------===//
// Statistics Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, Statistics) {
  factory->registerType("Type1", []() -> void* { return new BaseComponent(); });
  factory->registerType("Type2", []() -> void* { return new BaseComponent(); });

  factory->createByName("Type1");
  factory->createByName("Type2");

  auto& stats = factory->getStatistics();
  EXPECT_EQ(stats.typesRegistered, 2u);
  EXPECT_EQ(stats.instancesCreated, 2u);
}

//===----------------------------------------------------------------------===//
// Reset Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, Reset) {
  factory->registerType("TestType", []() -> void* { return new BaseComponent(); });
  factory->setTypeOverride("TestType", "Override");
  factory->setInstOverride("path", "TestType", "Override");

  factory->reset();

  EXPECT_FALSE(factory->isTypeRegistered("TestType"));
  EXPECT_EQ(factory->getTypeOverrides().size(), 0u);
  EXPECT_EQ(factory->getInstOverrides().size(), 0u);
}

//===----------------------------------------------------------------------===//
// Print Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, Print) {
  factory->registerType("TestType", []() -> void* { return new BaseComponent(); });
  factory->setTypeOverride("Orig", "Override");
  factory->setInstOverride("path", "Orig", "InstOverride");

  std::string output;
  llvm::raw_string_ostream os(output);
  factory->print(os);

  EXPECT_TRUE(output.find("TestType") != std::string::npos);
  EXPECT_TRUE(output.find("Type Overrides") != std::string::npos);
  EXPECT_TRUE(output.find("Instance Overrides") != std::string::npos);
}

//===----------------------------------------------------------------------===//
// UVMFactoryOverrideGuard Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMFactoryTest, OverrideGuardTypeOverride) {
  factory->registerType("Base", []() -> void* { return new BaseComponent(); });
  factory->registerType("Derived", []() -> void* { return new DerivedComponent(); });

  {
    UVMFactoryOverrideGuard guard(*factory, "Base", "Derived");

    void* obj = factory->createByName("Base");
    EXPECT_EQ(static_cast<BaseComponent*>(obj)->getTypeName(), "DerivedComponent");
    delete static_cast<BaseComponent*>(obj);
  }

  // After guard is destroyed, override should be removed
  void* obj = factory->createByName("Base");
  EXPECT_EQ(static_cast<BaseComponent*>(obj)->getTypeName(), "BaseComponent");
  delete static_cast<BaseComponent*>(obj);
}

//===----------------------------------------------------------------------===//
// Singleton Tests
//===----------------------------------------------------------------------===//

TEST(UVMFactorySingletonTest, GetInstance) {
  UVMFactory& factory1 = UVMFactory::getInstance();
  UVMFactory& factory2 = UVMFactory::getInstance();

  EXPECT_EQ(&factory1, &factory2);
}
