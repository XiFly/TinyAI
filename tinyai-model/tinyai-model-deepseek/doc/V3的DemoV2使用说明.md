# DeepSeek-V3 完整训练演示 V2 使用说明

## 概述

`DeepSeekV3TrainDemoV2` 是参考 `GPT1TrainDemoV2` 实现的完整训练流程演示，专为教学和学习场景设计。它展示了DeepSeek-V3模型从数据准备、预训练、后训练到推理的完整流程。

## 核心特性

### 1. 真实数据集生成
不同于 `DeepSeekV3TrainDemo` 使用虚拟随机数据，V2版本生成了真实的教学文本数据：

- **预训练数据集** (~250条): 
  - MoE和专家系统知识 (50条)
  - DeepSeek和大模型知识 (50条)
  - 任务感知和推理知识 (40条)
  - 代码生成和编程知识 (40条)
  - Transformer架构知识 (40条)
  - 深度学习基础知识 (30条)

- **后训练数据集** (~100条):
  - 带任务标签的指令-回答对
  - 支持5种任务类型: REASONING, CODING, MATH, GENERAL, MULTIMODAL
  - 覆盖MoE、DeepSeek、编程、Transformer、深度学习等主题

- **代码专项后训练数据集** (~60条):
  - 纯CODING任务的指令-回答对
  - 支持Python, Java, JavaScript, C++等语言
  - 用于强化MoE专家对代码任务的特化能力

### 2. 完整训练流程

#### 阶段0: 数据准备
```
./data/deepseek_v3_training/
├── pretrain.txt                # 预训练数据 (~250条)
├── posttrain_train.txt         # 通用后训练训练集 (~100条)
├── posttrain_val.txt           # 通用后训练验证集 (~15条)
├── code_posttrain_train.txt    # 代码专项训练集 (~60条)
└── code_posttrain_val.txt      # 代码专项验证集 (~10条)
```

#### 阶段1: 预训练 (Pretrain)
- **目标**: 学习语言的通用表示和MoE路由
- **任务**: 因果语言建模 + MoE负载均衡
- **数据**: 大规模无标注文本
- **学习率**: 2.5e-4
- **轮次**: 3 epochs
- **特色**: 稀疏激活(25%参数) + 专家网络

#### 阶段2: 通用后训练 (Posttrain)
- **目标**: 适应任务特定的推理和生成
- **任务**: 任务感知的指令跟随
- **数据**: 带任务标签的指令数据（混合任务）
- **学习率**: 2.5e-5 (比预训练低10倍)
- **轮次**: 3 epochs (带早停机制)
- **特色**: 任务路由 + 多任务适应

#### 阶段2B: 代码生成专项后训练 (Code Posttrain) **NEW!**
- **目标**: 强化MoE专家对代码任务的特化能力
- **任务**: 纯CODING任务的指令跟随
- **数据**: 60条代码生成问答（Python/Java/JS/C++）
- **学习率**: 1e-5 (比通用后训练更小)
- **轮次**: 4 epochs (代码任务需要更多轮次)
- **特色**: 持续激活CODING专家 → 专家特化能力增强
- **MoE优势**: 
  - 通用后训练：CODING数据仅占~20%，专家激活模式混合
  - 代码专项训练：CODING数据100%，持续强化特定专家
  - 预期效果：Expert 2,5成为代码专家，CODING任务时激活概率大幅提升

#### 阶段3: 推理 (Inference)
- **策略**: Greedy/Temperature/Top-K/Top-P
- **任务感知**: 根据任务类型自动路由专家
- **输出**: 生成的完整文本序列

### 3. 任务感知标注

后训练数据使用任务标签前缀：

```
[REASONING] Question: ... Answer: ...
[CODING] Question: ... Answer: ...
[MATH] Question: ... Answer: ...
[GENERAL] Question: ... Answer: ...
[MULTIMODAL] Question: ... Answer: ...
```

## 使用方法

### 运行完整演示

```bash
# 进入项目目录
cd /path/to/TinyAI

# 运行演示
mvn exec:java -pl tinyai-model/tinyai-model-deepseek \
  -Dexec.mainClass="io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3TrainDemoV2"
```

### 推荐JVM参数

由于模型训练需要一定内存，建议配置：

```bash
mvn exec:java -pl tinyai-model/tinyai-model-deepseek \
  -Dexec.mainClass="io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3TrainDemoV2" \
  -Dexec.args="" \
  -Dexec.cleanupDaemonThreads=false \
  -Xms512m -Xmx4g
```

## 数据集规模

### 预训练数据集
- **样本数**: ~250条
- **序列长度**: 32 tokens
- **批次大小**: 4
- **词汇表大小**: 自动构建（预计1000-2000词）

### 通用后训练数据集
- **训练集**: ~100条
- **验证集**: ~15条
- **序列长度**: 32 tokens
- **批次大小**: 2 (训练), 1 (验证)
- **任务分布**: 混合5种任务类型

### 代码专项后训练数据集 **NEW!**
- **训练集**: ~60条
- **验证集**: ~10条
- **序列长度**: 32 tokens
- **批次大小**: 2 (训练), 1 (验证)
- **任务类型**: 100% CODING任务
- **支持语言**: Python, Java, JavaScript, C++

### 内存占用
使用Micro配置（教学专用）：
- **隐藏维度**: 64
- **层数**: 2
- **专家数量**: 8
- **激活专家**: 2 (25%激活率)
- **预估内存**: < 2GB

## 输出示例

### 步骤0: 数据准备
```
📦 步骤0: 准备训练数据集
✓ 创建数据目录
✓ 预训练数据: 250 条
✓ 后训练训练集: 100 条
✓ 后训练验证集: 15 条
✓ 代码专项训练集: 60 条
✓ 代码专项验证集: 10 条
```

### 步骤1: 预训练
```
📚 步骤1: DeepSeek-V3 预训练
✓ 完整词汇表大小: 1523
✓ 模型配置: Micro
✓ 专家数量: 8, Top-K: 2
Epoch 1/3 | Step 50 | LM Loss: 6.2341 | MoE Loss: 0.001234 | ...
```

### 步骤2: 通用后训练
```
🎯 步骤2: DeepSeek-V3 后训练/微调
✓ 训练集: 100 条
✓ 任务感知标注: 启用
Epoch 1/3 | Step 20 | Loss: 5.8732 | 代码质量: 0.7234 | ...
```

### 步骤2B: 代码专项后训练 **NEW!**
```
💻 步骤2B: DeepSeek-V3 代码生成专项后训练
💡 目标：强化MoE专家对代码任务的特化能力
💡 策略：纯代码任务数据 + 更小学习率 + 更多训练轮次
✓ 代码专项训练集: 60 条
✓ 任务类型: 纯CODING (所有数据都是代码任务)
✓ 支持语言: Python, Java, JavaScript, C++
Epoch 1/4 | Step 15 | Loss: 4.9876 | 代码质量: 0.8456 | ...

✅ 代码专项后训练完成!
ℹ️ MoE专家特化说明:
  - 通用后训练: CODING数据仅占~20%, 专家激活模式混合
  - 代码专项训练: CODING数据100%, 持续强化特定专家
  - 预期效果: Expert 2,5成为代码专家, CODING任务时激活概率大幅提升
```

### 步骤3: 推理
```
🚀 步骤3: DeepSeek-V3 推理与文本生成
测试 1: "Mixture of Experts is"
  策略1 [Greedy贪婪]: 
    → mixture of experts is a neural network architecture that uses ...
  策略2 [Temperature=0.8]: 
    → mixture of experts is an architecture that combines multiple ...
```

## 与GPT1TrainDemoV2的对比

| 特性 | GPT1TrainDemoV2 | DeepSeekV3TrainDemoV2 |
|------|-----------------|----------------------|
| 模型架构 | Transformer Decoder | MoE + Task-aware |
| 专家系统 | ❌ | ✅ (8专家, Top-2) |
| 任务感知 | ❌ | ✅ (5种任务类型) |
| 稀疏激活 | ❌ | ✅ (25%激活率) |
| 负载均衡 | ❌ | ✅ (MoE负载损失) |
| 代码质量评估 | ❌ | ✅ (4维评分) |
| 数据规模 | ~500条 | ~350条 |
| 词汇表 | ~2000词 | ~1500词 |

## 关键代码组件

### 简单分词器
```java
SimpleTokenizer sharedTokenizer = new SimpleTokenizer();
List<Integer> tokens = sharedTokenizer.encode("Mixture of Experts is");
String text = sharedTokenizer.decode(tokens);
```

### 任务标签处理
```java
// 提取任务类型
TaskType taskType = extractTaskType("[REASONING] Question: ...");

// 移除任务标签
String cleanText = removeTaskLabel("[REASONING] Question: ...");
```

### 数据集创建
```java
DeepSeekV3Dataset dataset = createDatasetFromTexts(
    texts,
    maxSeqLength,
    batchSize,
    vocabSize,
    useTaskLabels  // true表示使用任务标签
);
```

## 扩展和定制

### 1. 增加数据集规模

修改数据生成函数，添加更多文本：

```java
private static List<String> generateMoETexts() {
    return Arrays.asList(
        // 添加更多MoE相关文本...
    );
}
```

### 2. 调整模型配置

使用不同的模型规模：

```java
// 更大的模型 (需要更多内存)
DeepSeekV3Config config = DeepSeekV3Config.createSmallConfig();

// 极小模型 (最省内存)
DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
```

### 3. 修改训练超参数

```java
trainer.configure(
    5,          // maxEpochs - 增加训练轮次
    1e-4f,      // learningRate - 调整学习率
    100,        // warmupSteps - 调整warmup步数
    0.5f        // maxGradNorm - 调整梯度裁剪
);
```

### 4. 添加新的任务类型

在数据生成函数中添加新的任务标签：

```java
qa.add("[CUSTOM_TASK] Question: ... Answer: ...");
```

## 常见问题

### Q1: 内存不足怎么办？
A: 使用Micro配置并增加JVM堆内存：`-Xmx4g`

### Q2: 训练速度慢怎么办？
A: 减少数据规模或训练轮次，使用更小的batch size

### Q3: 如何使用自己的数据？
A: 修改数据生成函数，或直接编辑生成的txt文件

### Q4: 如何保存和加载模型？
A: 训练过程会自动保存检查点到 `./checkpoints/deepseek_v3_v2/`

### Q5: 推理结果不理想怎么办？
A: 增加训练数据规模和轮次，调整学习率，使用更大的模型配置

## 学习建议

1. **先运行完整流程**: 了解整体架构和训练过程
2. **研究数据生成**: 理解不同任务类型的数据特点
3. **观察训练日志**: 关注损失变化和MoE负载均衡
4. **实验不同策略**: 尝试不同的推理生成策略
5. **对比V1版本**: 理解真实数据和虚拟数据的差异

## 参考文档

- `GPT1TrainDemoV2.java` - 原始实现参考
- `DeepSeekV3TrainDemo.java` - V1版本（虚拟数据）
- `DeepSeekV3Pretrain.java` - 预训练器实现
- `DeepSeekV3Posttrain.java` - 后训练器实现
- `DeepSeekV3Inference.java` - 推理引擎实现
- `DeepSeekV3Dataset.java` - 数据集实现

## 总结

`DeepSeekV3TrainDemoV2` 提供了一个完整、可运行的DeepSeek-V3模型训练演示，特别适合：

- ✅ 学习MoE架构的训练流程
- ✅ 理解任务感知的数据标注和训练
- ✅ 体验从预训练到后训练的完整周期
- ✅ 实验不同的生成策略
- ✅ 作为自定义训练的起点

通过这个演示，可以深入理解DeepSeek-V3的核心特性：MoE稀疏激活、任务感知路由、负载均衡训练等。
