# TinyAI Banana - 多模态图像生成模型

> 基于 TinyAI 框架实现的 Gemini Nano Banana 风格多模态图像生成模型

## 📚 项目概述

`tinyai-model-banana` 是 TinyAI 项目中对 Google Gemini Nano Banana（Gemini 2.5 Flash Image）的教育性实现。该模块基于 TinyAI V2 架构，实现了文本-图像多模态融合和图像生成能力。

### 设计理念

**Banana 的核心价值在于「教育友好的多模态学习」**:

- **🎓 教育友好**: 清晰的多模态架构设计，完整的文档，适合学习 Vision Transformer 和多模态融合
- **💡 轻量化设计**: 针对教育场景优化，可在普通 GPU 上训练和推理
- **🔧 功能完整**: 支持文本编码、图像编码、跨模态注意力、图像生成等完整流程
- **🚀 纯 Java 实现**: 基于 TinyAI 框架，易于集成和扩展

### 核心特性

| 特性类别 | 功能说明 |
|---------|----------|
| **模型架构** | Vision Transformer · 跨模态注意力 · Patch嵌入 · 2D位置编码 |
| **编码能力** | 文本编码器 · 图像编码器 · 多模态融合 |
| **生成能力** | 文本到图像 · 图像编辑 · 图像理解 |
| **工程特性** | 纯 Java 实现 · V2 组件架构 · 复用Conv2D算子 |

## 🏗️ 架构设计

### 整体架构

Banana 采用多模态 Transformer 架构：

```
输入层: 文本提示词 + 图像输入
   ↓
编码层: TextEncoder + ImageEncoder (ViT)
   ↓
融合层: 跨模态注意力 (CrossModalAttention)
   ↓
生成层: ImageDecoder (自回归/扩散)
   ↓
输出层: 生成图像 + 质量评估
```

### 模块结构

```
tinyai-model-banana/
├── src/main/java/io/leavesfly/tinyai/banana/
│   ├── config/                          # 配置管理
│   │   ├── BananaConfig.java               # 模型配置
│   │   └── TaskType.java                   # 任务类型枚举
│   │
│   ├── encoder/                         # 编码器模块
│   │   ├── TextEncoder.java                # 文本编码器
│   │   ├── ImageEncoder.java               # 图像编码器(ViT)
│   │   ├── PatchEmbedding.java             # Patch嵌入
│   │   └── Position2D.java                 # 2D位置编码
│   │
│   ├── transformer/                     # Transformer核心
│   │   ├── BananaTransformerBlock.java     # 多模态Transformer块
│   │   ├── MultiModalAttention.java        # 多模态注意力
│   │   └── CrossModalAttention.java        # 跨模态注意力
│   │
│   ├── decoder/                         # 解码器模块(待实现)
│   │   ├── ImageDecoder.java               # 图像解码器
│   │   └── ImageTokenizer.java             # 图像Tokenizer
│   │
│   ├── block/                           # 主体块
│   │   └── BananaBlock.java                # 模型主体
│   │
│   ├── model/                           # 模型类
│   │   └── BananaModel.java                # 模型接口(继承Model)
│   │
│   ├── training/                        # 训练组件
│   │   ├── dataset/
│   │   │   └── BananaDataset.java          # 多模态数据集
│   │   ├── PretrainTrainer.java            # 预训练器
│   │   ├── FinetuneTrainer.java            # 微调器
│   │   └── demo/
│   │       └── TrainDemo.java              # 训练演示
│   │
│   └── demo/                            # 演示程序
│       └── BananaDemo.java                 # 推理演示
│
└── README.md                             # 本文档
```

## 📊 配置规模

### 预设配置对比

| 配置项 | Tiny (教学) | Small (实验) | Base (标准) |
|--------|------------|-------------|------------|
| **参数量** | 60.82M | 166.72M | 385.88M |
| **隐藏维度** | 512 | 768 | 1024 |
| **层数** | 8 | 12 | 16 |
| **注意力头数** | 8 | 12 | 16 |
| **FFN维度** | 2048 | 3072 | 4096 |
| **图像尺寸** | 256x256 | 384x384 | 512x512 |
| **Patch尺寸** | 16x16 | 16x16 | 16x16 |
| **Patch数量** | 256 | 576 | 1024 |
| **图像编码器层数** | 6 | 9 | 12 |

## 🚀 快速开始

### 1. 创建模型实例

```java
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

// 方式1: 使用预设配置
BananaModel model = BananaModel.create("banana_tiny", "tiny");
System.out.println(model.getConfigSummary());

// 方式2: 自定义配置
BananaConfig config = new BananaConfig();
config.setHiddenSize(768);
config.setNumLayers(12);
config.setImageSize(384);
config.updateNumPatches();
BananaModel customModel = new BananaModel("banana_custom", config);
```

### 2. 文本编码

```java
// 准备文本输入 [batch, seq_len]
NdArray textData = NdArray.of(Shape.of(2, 10));  // 2个样本，长度10
Variable textInput = new Variable(textData);

// 文本编码
Variable textFeatures = model.encodeText(textInput);
System.out.println("文本特征: " + textFeatures.getValue().getShape());
// 输出: [2, 10, 512]
```

### 3. 图像编码

```java
// 准备图像输入 [batch, channels, height, width]
NdArray imageData = NdArray.of(Shape.of(2, 3, 256, 256));
// 随机初始化
float[] array = imageData.getArray();
for (int i = 0; i < array.length; i++) {
    array[i] = (float) Math.random();
}
Variable imageInput = new Variable(imageData);

// 图像编码
Variable imageFeatures = model.encodeImage(imageInput);
System.out.println("图像特征: " + imageFeatures.getValue().getShape());
// 输出: [2, 256, 512]  (256个patches，每个512维)
```

### 4. 多模态融合

```java
import io.leavesfly.tinyai.banana.config.TaskType;

// 文本和图像编码
Variable textFeatures = model.encodeText(textInput);
Variable imageFeatures = model.encodeImage(imageInput);

// 跨模态融合
Variable fusedResult = model.getBananaBlock().forwardMultiModal(
    textFeatures, 
    imageFeatures, 
    TaskType.TEXT_TO_IMAGE
);

System.out.println("融合结果: " + fusedResult.getValue().getShape());
```

## 🎯 训练流程

### 预训练

```java
import io.leavesfly.tinyai.banana.training.PretrainTrainer;
import io.leavesfly.tinyai.banana.training.dataset.BananaDataset;

// 1. 创建模型
BananaModel model = BananaModel.create("banana_tiny", "tiny");

// 2. 准备数据
BananaDataset dataset = new BananaDataset(32, 256, 4);
dataset.loadSyntheticData(1000);  // 合成数据演示
// dataset.loadFromCSV("data/train.csv");  // 真实数据

// 3. 配置训练器
PretrainTrainer trainer = new PretrainTrainer(model, dataset);
trainer.configure(
    10,      // epochs
    1e-3f,   // learningRate
    0,       // warmupSteps
    1.0f     // maxGradNorm
);
trainer.setCheckpoint("./checkpoints/banana_pretrain", 100);

// 4. 开始训练
trainer.train();
```

### 微调

```java
import io.leavesfly.tinyai.banana.training.FinetuneTrainer;

// 1. 准备训练和验证数据
BananaDataset trainDataset = new BananaDataset(32, 256, 4);
trainDataset.loadSyntheticData(200);

BananaDataset valDataset = new BananaDataset(32, 256, 4);
valDataset.loadSyntheticData(50);

// 2. 配置微调器
FinetuneTrainer finetuner = new FinetuneTrainer(model, trainDataset, valDataset);
finetuner.configure(
    5,       // epochs
    1e-4f,   // learningRate (比预训练小10倍)
    3        // patience (早停)
);

// 3. 开始微调
finetuner.train();
System.out.println("最佳验证损失: " + finetuner.getBestValLoss());
```

### 运行训练演示

```bash
# 完整训练流程(预训练 + 微调)
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.banana.training.demo.TrainDemo"

# 只运行预训练
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.banana.training.demo.TrainDemo" \
    -Dexec.args="pretrain"

# 只运行微调
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.banana.training.demo.TrainDemo" \
    -Dexec.args="finetune"
```

更多详细信息请参考 **[训练指南](doc/训练指南.md)**。

## 📦 依赖项

| 依赖模块 | 说明 | 用途 |
|---------|------|------|
| **tinyai-deeplearning-ml** | 机器学习核心 | Model基类、Trainer |
| **tinyai-deeplearning-nnet** | 神经网络层 | Conv2D、Linear、LayerNorm等 |
| **tinyai-deeplearning-func** | 自动微分引擎 | Variable、Function |
| **tinyai-deeplearning-ndarr** | 多维数组 | NdArray基础运算 |

## ✅ 开发进度

### 阶段一：基础架构 (✅ 已完成)

- [x] 模块创建和pom配置
- [x] BananaConfig配置类
- [x] TaskType任务类型
- [x] BananaBlock主体框架
- [x] BananaModel模型类
- [x] TextEncoder文本编码器
- [x] 基础编译测试

### 阶段二：图像编码器 (✅ 已完成)

- [x] PatchEmbedding实现
- [x] Position2D位置编码
- [x] ImageEncoder实现
- [x] 图像预处理工具
- [x] 编码器测试

### 阶段三：跨模态注意力 (✅ 已完成)

- [x] CrossModalAttention实现
- [x] MultiModalFusion实现
- [x] 注意力掩码生成
- [x] 特征融合测试

### 阶段四：训练框架 (✅ 已完成)

- [x] BananaDataset数据集
- [x] PretrainTrainer预训练器
- [x] FinetuneTrainer微调器
- [x] TrainDemo训练演示
- [x] 编译测试

## 🔬 技术亮点

### 1. 复用现有Conv2D算子

```java
// 使用TinyAI已优化的Conv2D
import io.leavesfly.tinyai.nnet.v2.layer.conv.Conv2d;

// Patch嵌入中使用卷积
Conv2d patchConv = new Conv2d(
    "patch_conv", 
    imageChannels,    // 输入通道:3(RGB)
    hiddenSize,       // 输出通道:512
    patchSize,        // 卷积核大小:16
    patchSize,        // 步长:16(无重叠)
    0                 // 无padding
);
```

### 2. Variable层面计算

所有操作基于Variable，支持完整的自动微分：

```java
Variable patchEmbeddings = patchConv.forward(imageInput);
// 支持梯度回传
patchEmbeddings.backward();
```

### 3. 模块化设计

遵循TinyAI的Block-Layer分层设计模式，便于复用和扩展。

## 📚 文档资料

| 文档 | 说明 | 链接 |
|------|------|------|
| **技术架构文档** | 系统架构、核心组件、关键实现 | [doc/技术架构文档.md](doc/技术架构文档.md) |
| **API参考文档** | 完整的API接口说明和使用示例 | [doc/API参考文档.md](doc/API参考文档.md) |
| **训练指南** | 预训练、微调、最佳实践 | [doc/训练指南.md](doc/训练指南.md) |

## 📚 相关资源

### 论文
- Vision Transformer (ViT): "An Image is Worth 16x16 Words"
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision"
- Flamingo: "Tackling Multiple Tasks with a Single Visual Language Model"

### TinyAI相关模块
- [GPT-1模型](../tinyai-model-gpt/) - 文本生成模型
- [MiniMind模型](../tinyai-model-minimind/) - 轻量级语言模型
- [DeepSeek模型](../tinyai-model-deepseek/) - 深度思考模型

## 📝 开发日志

### 2025-12-21
- ✅ 创建tinyai-model-banana模块
- ✅ 实现BananaConfig配置类(三种预设配置)
- ✅ 实现TaskType任务枚举
- ✅ 实现BananaBlock主体框架
- ✅ 实现BananaModel模型类
- ✅ 实现TextEncoder文本编码器
- ✅ 实现PatchEmbedding切片嵌入
- ✅ 实现Position2D位置编码
- ✅ 实现ImageEncoder图像编码器
- ✅ 实现CrossModalAttention跨模态注意力
- ✅ 实现MultiModalFusion多模态融合
- ✅ 编写BananaDemo演示程序
- ✅ 编译测试通过
- ✅ 功能验证成功
- ✅ 编写技术文档
- ✅ 编写API参考文档
- ✅ 创建BananaDataset数据集类
- ✅ 创建PretrainTrainer预训练器
- ✅ 创建FinetuneTrainer微调器
- ✅ 创建TrainDemo训练演示
- ✅ 训练框架编译验证成功

---

**项目状态**: 🎉 **阶段一二三四全部完成** - 核心功能+训练框架已实现

**项目统计**:
- 💻 代码文件: 18个 Java文件
- 📋 代码量: ~4100 行
- 📦 模块分类: config(2) + encoder(4) + fusion(2) + block(1) + model(1) + training(4) + demo(2) + doc(2)
- 💡 API覆盖: 配置管理 + 编码器 + 融合层 + 训练器 + 模型接口

**下一步**: 实现图像解码器，支持文本生成图像
