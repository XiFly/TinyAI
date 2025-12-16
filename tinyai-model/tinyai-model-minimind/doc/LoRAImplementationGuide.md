# LoRA实现指南

## 概述

本模块提供了LoRA(Low-Rank Adaptation)微调的**配置**和**训练器**实现。

由于完整的LoRA实现需要修改现有模型架构(在Linear层中注入低秩矩阵),目前提供的是基础框架。

## 已实现组件

### 1. LoRAConfig.java
- LoRA配置类
- 包含rank、alpha、dropout、目标模块等配置
- 提供默认配置和全量注意力配置

### 2. LoRATrainer.java  
- LoRA训练器
- 通过参数名称过滤,仅训练包含"lora"的参数
- 冻结其他参数
- 支持梯度裁剪、检查点保存

## LoRA原理

LoRA通过在原始权重矩阵旁添加低秩分解矩阵:

```
output = W * x + (B * A) * x * (alpha / rank)
```

其中:
- `W`: 原始权重(冻结)
- `A`: 低秩矩阵 [in_features, rank]  
- `B`: 低秩矩阵 [rank, out_features]
- `alpha / rank`: 缩放因子

## 完整实现所需步骤

### 方案1: 修改MultiHeadAttention

1. 在`MultiHeadAttention.java`中为Q、K、V、O投影层添加LoRA参数
2. 在forward时计算LoRA路径并与原始输出相加
3. 注册LoRA参数,参数名包含"lora"关键字

### 方案2: 创建LoRA包装器

1. 创建`LoRAWrapper`类包装现有Linear层
2. 动态添加LoRA参数
3. 重写forward方法

### 方案3: 使用外部LoRA库

1. 参考HuggingFace PEFT库的Java实现
2. 通过适配器模式集成

## 当前使用方式

虽然未完全实现LoRA层,但可以通过以下方式使用:

```java
// 1. 创建LoRA配置
LoRAConfig loraConfig = LoRAConfig.createDefault();

// 2. 手动标记需要微调的参数(在参数名中包含"lora")
// 例如在模型中添加额外的可学习参数

// 3. 使用LoRATrainer训练
LoRATrainer trainer = new LoRATrainer(model, dataset, loraConfig);
trainer.configure(3, 1e-4f, 1.0f);
trainer.train();
```

## 限制

1. **未实现LoRALinear层**: 需要对V2 Module进行扩展
2. **参数效率**: 当前实现无法达到真正的参数高效(90%+减少)
3. **权重合并**: 未实现LoRA权重合并到基础模型

## 替代方案

在完整LoRA实现之前,可以使用:

1. **SFT微调**: 使用较小学习率微调全部参数
2. **选择性微调**: 仅微调部分层(如最后几层)
3. **AdapterAnother lightweight tuning method

## 未来改进

- [ ] 实现完整的LoRALinear层
- [ ] 支持动态注入LoRA到现有模型
- [ ] 实现权重合并功能
- [ ] 支持多适配器切换
- [ ] 添加LoRA权重保存/加载

---

**注**: 当前实现主要用于理解LoRA原理和训练流程,实际使用建议使用SFT进行微调。
