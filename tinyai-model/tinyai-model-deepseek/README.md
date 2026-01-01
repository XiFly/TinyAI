# DeepSeek 模型实现

基于 TinyAI 框架**完全独立**实现的 DeepSeek 系列大语言模型，包含 DeepSeek-V3 和 DeepSeek-R1 两个主力模型。100% 基于 **nnet v2 API**，引入混合专家模型(MoE)、推理增强、反思机制等前沿技术，支持代码生成、数学推理、多任务处理等能力。

## ✨ 核心特点

- ✅ **完全独立实现** - 100% 基于 V2 API，零依赖旧版组件
- ✅ **双模型支持** - DeepSeek-V3(MoE) + DeepSeek-R1(推理增强)
- ✅ **混合专家架构** - 8专家网络，Top-2路由，任务感知选择
- ✅ **推理增强** - 多步推理、思维链生成、自我反思机制
- ✅ **代码生成优化** - 支持10种编程语言，质量评估系统
- ✅ **Variable层面计算** - 完整计算图，梯度正确回传
- ✅ **完整文档** - 详细的代码注释和架构说明

## 📁 文件结构

```
tinyai-model-deepseek/
├── src/main/java/io/leavesfly/tinyai/deepseek/
│   ├── v3/                                # DeepSeek-V3 (MoE)
│   │   ├── DeepSeekV3Config.java          # 完全独立配置类（683行）
│   │   ├── DeepSeekV3TokenEmbedding.java  # ✅ Variable层面（indexSelect/reshape/repeat）
│   │   ├── DeepSeekV3TransformerBlock.java # V2 Module
│   │   ├── DeepSeekV3MoELayer.java        # ✅ 批量专家计算，完整Variable层面
│   │   ├── DeepSeekV3ReasoningBlock.java  # 任务感知推理
│   │   ├── DeepSeekV3CodeBlock.java       # 代码生成（10种语言）
│   │   ├── DeepSeekV3Block.java           # 主体块
│   │   ├── DeepSeekV3Model.java           # 模型类
│   │   ├── DeepSeekV3Demo.java            # 演示程序
│   │   ├── TaskType.java                  # 5种任务类型
│   │   └── training/                      # 训练器（Pretrain/Finetune/RL/Inference/Evaluator）
│   └── r1/                                # DeepSeek-R1 (推理增强)
│       ├── DeepSeekR1Config.java          # 完全独立配置类（481行）
│       ├── DeepSeekR1TokenEmbedding.java  # ✅ Variable层面（indexSelect/reshape/repeat）
│       ├── DeepSeekR1TransformerBlock.java # V2 Module
│       ├── DeepSeekR1ReasoningBlock.java  # 7步迭代推理
│       ├── DeepSeekR1ReflectionBlock.java # 自我评估与改进
│       ├── DeepSeekR1Block.java           # 主体块
│       ├── DeepSeekR1Model.java           # 模型类
│       ├── DeepSeekR1Demo.java            # 演示程序
│       └── training/                      # 训练器（Pretrain/Finetune/RL/Inference/Evaluator/Generator）
└── README.md
```

**总代码量**: 
- **DeepSeek-V3**: ~3,500行，100% V2 API，完整Variable层面
- **DeepSeek-R1**: ~2,800行，100% V2 API，完整Variable层面

## 🎯 模型对比

### DeepSeek-V3 vs DeepSeek-R1

| 特性 | DeepSeek-V3 (MoE) | DeepSeek-R1 (推理增强) |
|------|-------------------|---------------------|
| 架构 | 混合专家模型(8专家,Top-2) | 标准Transformer |
| 主要能力 | 代码生成、多任务处理 | 推理、反思、思维链 |
| 任务感知 | ✅ 5种任务类型路由 | ❌ 通用推理 |
| 专家网络 | ✅ 8专家，动态选择 | ❌ 无 |
| 推理步骤 | 任务适应性推理 | ✅ 7步迭代推理 |
| 反思机制 | ✅ 自我纠错 | ✅ 完整反思模块 |
| 代码生成 | ✅ 10种语言，质量评估 | ❌ 通用生成 |
| 数学推理 | ✅ 专用数学处理 | ✅ 通用推理 |
| 参数效率 | ✅ 激活~25%参数 | ❌ 全部激活 |
| 适用场景 | 代码、数学、多模态 | 推理、问题求解 |

## 🚀 快速开始

### 环境要求

- **Java**: JDK 8+
- **Maven**: 3.6+
- **内存**: 推荐 4GB+
- **依赖**: TinyAI 核心模块

### 1. DeepSeek-V3 基本使用

```java
import io.leavesfly.tinyai.deepseek.v3.*;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

// 1. 创建不同规模的V3模型
DeepSeekV3Model tinyModel = DeepSeekV3Model.createTinyModel("v3-tiny");      // 快速测试
DeepSeekV3Model standardModel = DeepSeekV3Model.createStandardModel("v3-std"); // 标准配置
DeepSeekV3Model largeModel = DeepSeekV3Model.createLargeModel("v3-large");   // 大型模型

// 2. 打印模型信息
standardModel.printModelInfo();

// 3. 基础推理
NdArray tokenIds = NdArray.of(new int[][]{{1, 15, 23, 42}});
Variable input = new Variable(tokenIds);
Variable output = standardModel.forward(input);
System.out.println("输出形状: " + output.getValue().getShape());

// 4. 代码生成（任务感知）
NdArray codePrompt = createCodePrompt(); // 代码提示
Variable codeOutput = standardModel.forward(new Variable(codePrompt));
```

### 2. DeepSeek-R1 基本使用

```java


// 1. 创建不同规模的R1模型
DeepSeekR1Model tinyModel=DeepSeekR1Model.createTinyModel("r1-tiny");      // 快速测试
        DeepSeekR1Model standardModel=DeepSeekR1Model.createStandardModel("r1-std"); // 标准配置
        DeepSeekR1Model largeModel=DeepSeekR1Model.createLargeModel("r1-large");   // 大型模型

// 2. 打印模型信息
        standardModel.printModelInfo();

// 3. 基础推理
        NdArray tokenIds=NdArray.of(new int[][]{{1,15,23,42}});
        Variable input=new Variable(tokenIds);
        Variable output=standardModel.forward(input);

// 4. 带反思的推理
        DeepSeekR1Block.ReasoningOutput reasoningOutput=standardModel.forwardWithReasoning(input);
        System.out.println("推理质量: "+reasoningOutput.getQualityScore());
        System.out.println("需要改进: "+reasoningOutput.needsRefinement());
```

### 3. 自定义配置

```java
// V3自定义配置
DeepSeekV3Config v3Config = new DeepSeekV3Config();
v3Config.setVocabSize(50257);
v3Config.setNEmbd(768);
v3Config.setNLayer(12);
v3Config.setNHead(12);
v3Config.setNumExperts(8);           // 8个专家
v3Config.setTopK(2);                  // Top-2选择
v3Config.setEnableTaskAwareRouting(true); // 启用任务感知
DeepSeekV3Model customV3 = new DeepSeekV3Model("custom-v3", v3Config);

// R1自定义配置
DeepSeekR1Config r1Config = new DeepSeekR1Config();
r1Config.setVocabSize(50257);
r1Config.setNEmbd(512);
r1Config.setNLayer(6);
r1Config.setMaxReasoningSteps(7);    // 7步推理
r1Config.setConfidenceThreshold(0.7f); // 置信度阈值
DeepSeekR1Model customR1 = new DeepSeekR1Model("custom-r1", r1Config);
```

## 🔍 核心优势

### 1. 完全独立的V2架构

**DeepSeekV3Config** - 完全独立配置类（683行）
- ✅ 零依赖旧配置，所有参数独立定义
- ✅ MoE配置：numExperts、topK、loadBalanceLossWeight等
- ✅ 任务感知配置：taskEmbedDim、numTaskTypes等
- ✅ 代码生成配置：codeQualityDim、numProgrammingLanguages等
- ✅ 完整的Getter/Setter和validate()方法

**DeepSeekR1Config** - 完全独立配置类（481行）
- ✅ 零继承旧配置，所有参数独立定义
- ✅ 推理配置：maxReasoningSteps、confidenceThreshold等
- ✅ 反思配置：reflectionHiddenDim、qualityThreshold等
- ✅ 完整的Getter/Setter和validate()方法

### 2. 100% V2 API + Variable层面计算

**DeepSeekV3TokenEmbedding** - Token嵌入层（V2 Module）
- ✅ 完全基于V2 Module实现
- ✅ 使用V2 Parameter管理嵌入矩阵
- ✅ **完全在Variable层面**：使用`indexSelect`、`reshape`、`repeat`算子
- ✅ Token嵌入 + 位置嵌入 + Dropout
- ✅ **梯度完整回传**：从输出到嵌入参数的完整计算图

**DeepSeekV3MoELayer** - 混合专家层（V2 Module）
- ✅ 完全基于V2 Module实现
- ✅ **批量专家计算**：所有专家并行处理整个batch
- ✅ **Variable层面算子**：`add`、`mul`、`softMax`、`indexSelect`、`repeat`
- ✅ **完整计算图**：梯度可以正确回传到专家参数
- ✅ **核心突破**：解决了MoE动态路由的Variable化问题

**DeepSeekV3TransformerBlock** - Transformer块（V2 Module）
- ✅ 100%使用V2组件：LayerNorm、MultiHeadAttention、Linear、GELU、Dropout
- ✅ Pre-LayerNorm架构
- ✅ 因果掩码自动生成

**DeepSeekR1TokenEmbedding** - Token嵌入层（V2 Module）
- ✅ 与V3相同的Variable层面实现
- ✅ 使用`indexSelect`、`reshape`、`repeat`算子
- ✅ 完整计算图，梯度正确回传

### 3. 混合专家模型(MoE)的Variable化突破

**批量计算优化**：
- ✅ **所有专家并行**：8个专家同时处理整个batch
- ✅ **权重mask**：根据Top-2结果构建权重矩阵
- ✅ **Variable组合**：使用`mul`和`add`进行加权组合
- ✅ **梯度完整**：从输出到每个专家参数的完整计算图

**核心代码流程**：
```java
// 1. 所有专家并行处理整个batch
for (int i = 0; i < numExperts; i++) {
    expertOutputs.add(experts.get(i).forward(input));  // ✅ Variable层面
}

// 2. 构建权重mask并组合
for (int expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    Variable weightMask = createExpertWeightMask(expertIdx, topKResult);
    Variable weightMask3D = weightMask.repeat(1, 1, nEmbd);  // ✅ Variable.repeat
    Variable weightedOut = expertOut.mul(weightMask3D);       // ✅ Variable.mul
    output = output.add(weightedOut);                        // ✅ Variable.add
}
```

**任务感知路由**：
- ✅ **5种任务类型**：REASONING, CODING, MATH, GENERAL, MULTIMODAL
- ✅ **任务偏置**：不同任务倾向选择不同专家（使用Variable.add）
- ✅ **负载均衡**：确保所有专家被均匀使用

### 4. 推理增强能力

**DeepSeek-R1推理机制**：
- ✅ **7步迭代推理**：多步推理状态管理
- ✅ **置信度评估**：动态评估每步置信度
- ✅ **自我反思**：推理质量评估和改进建议
- ✅ **思维链生成**：输出完整的推理过程

**DeepSeek-V3推理机制**：
- ✅ **任务类型识别**：自动识别任务类型
- ✅ **专门化推理器**：针对不同任务的专用推理逻辑
- ✅ **自我纠错**：推理结果验证和纠正
- ✅ **置信度评估**：多维度置信度评估

## 📊 性能特点

### 模型规模对比

| 模型规模 | 参数量 | 层数 | 维度 | 头数 | 专家数 | 工厂方法 | V2组件 | Variable层面 |
|---------|-------|------|------|------|---------|------------|--------|------------|
| **V3-Tiny** | ~30M | 6 | 256 | 8 | 4 | createTinyModel() | ✅ 100% | ✅ 100% |
| **V3-Standard** | ~150M | 12 | 768 | 12 | 8 | createStandardModel() | ✅ 100% | ✅ 100% |
| **V3-Large** | ~500M | 24 | 1024 | 16 | 8 | createLargeModel() | ✅ 100% | ✅ 100% |
| **R1-Tiny** | ~20M | 6 | 256 | 8 | - | createTinyModel() | ✅ 100% | ✅ 100% |
| **R1-Standard** | ~100M | 12 | 512 | 8 | - | createStandardModel() | ✅ 100% | ✅ 100% |
| **R1-Large** | ~350M | 18 | 768 | 12 | - | createLargeModel() | ✅ 100% | ✅ 100% |

### V2组件使用情况

| 组件 | 类型 | 使用位置 | V2版本 |
|------|------|----------|--------|
| Module | 基类 | 所有层 | ✅ |
| Parameter | 参数管理 | Token/Position嵌入、专家网络 | ✅ |
| LayerNorm | 归一化 | Transformer块、最终层 | ✅ |
| MultiHeadAttention | 注意力 | Transformer块 | ✅ |
| Linear | 线性层 | 门控、MLP、输出投影、专家网络 | ✅ |
| GELU | 激活函数 | MLP、专家网络 | ✅ |
| Dropout | 正则化 | 所有分支 | ✅ |

### Variable层面算子使用

| 算子 | 用途 | 使用位置 | 状态 |
|------|------|----------|------|
| `indexSelect` | 索引选择嵌入 | TokenEmbedding | ✅ |
| `reshape` | 形状变换 | TokenEmbedding, MoELayer | ✅ |
| `repeat` | 维度重复 | TokenEmbedding, MoELayer | ✅ |
| `add` | 向量加法 | TokenEmbedding, MoELayer | ✅ |
| `mul` | 向量乘法 | MoELayer | ✅ |
| `softMax` | Softmax激活 | MoELayer | ✅ |

### 验证清单

✅ **零import旧组件** - 已验证  
✅ **零旧类引用** - 已验证  
✅ **零旧Config继承** - 已验证  
✅ **所有文件编译通过** - 已验证  
✅ **V2 API完整性** - 已验证  
✅ **Variable层面计算** - 已验证  
✅ **计算图完整性** - 已验证  
✅ **梯度正确回传** - 已验证

## 🧪 完整演示

运行演示程序查看完整功能：
- [DeepSeekV3Demo.java](src/main/java/io/leavesfly/tinyai/deepseek/v3/DeepSeekV3Demo.java)
- [DeepSeekR1Demo.java](src/main/java/io/leavesfly/tinyai/deepseek/rx/DeepSeekR1Demo.java)

## 📚 详细文档

- [DeepSeek-V3 详细实现说明](src/main/java/io/leavesfly/tinyai/deepseek/v3/README.md)
- [DeepSeek-R1 详细实现说明](src/main/java/io/leavesfly/tinyai/deepseek/rx/README.md)

## 🔧 高级特性

### 训练支持

每个模型都提供完整的训练支持：

- **预训练** (Pretrain): 从头训练模型
- **微调** (Finetune): 在预训练模型基础上进行微调
- **强化学习** (RL): 基于奖励的强化学习训练
- **评估** (Evaluation): 模型效果评估
- **推理** (Inference): 模型推理生成

### 支持的任务类型

**DeepSeek-V3**：
- ✅ **REASONING** - 推理任务
- ✅ **CODING** - 代码生成（10种编程语言）
- ✅ **MATH** - 数学计算
- ✅ **GENERAL** - 通用对话
- ✅ **MULTIMODAL** - 多模态处理

**DeepSeek-R1**：
- ✅ **通用推理任务**
- ✅ **思维链推理**
- ✅ **文本生成**
- ✅ **质量评估**

## 👏 致谢

感谢以下项目和团队的贡献：

- **DeepSeek 团队**: 提供了优秀的模型架构和实现参考
- **TinyAI 框架**: 提供了完整的深度学习基础设施
- **开源社区**: 提供了宝贵的意见和建议

---

<div align="center">
  <h3>🎯 让 DeepSeek 模型在 Java 生态中发光发热</h3>
  <p>如果这个模块对您有帮助，请给我们一个⭐️</p>
</div>
