# TinyAI 世界模型具身智能模块（World Model）

## 📖 模块简介

`tinyai-agent-embodied-wm` 是 TinyAI 智能体系统层的重要组成部分，专注于**基于世界模型的具身智能**（World Model-based Embodied Intelligence）技术实现。本模块实现了完整的世界模型架构，展示了智能体如何通过学习环境的内部表示来进行高效的端到端学习和规划。

### 核心特性

- 🧠 **完整的世界模型架构**：VAE编码器 + MDN-RNN记忆 + 控制器
- 🎯 **端到端学习能力**：从原始观察直接到动作决策
- 💭 **想象训练**：在内部模型中进行规划，无需真实环境交互
- 🔄 **高效样本利用**：通过想象rollout提高数据效率
- 🎨 **可扩展设计**：模块化架构，便于定制和扩展

## 🏗️ 系统架构

### 世界模型三大核心组件

```
世界模型架构
├── VAE编码器（Vision Component）
│   ├── 编码器：Observation -> (μ, σ²)
│   ├── 重参数化：z = μ + σ·ε
│   └── 解码器：z -> Reconstructed Observation
│
├── MDN-RNN（Memory Component）
│   ├── GRU单元：维护时序记忆
│   ├── 混合密度网络：预测下一状态分布
│   └── 采样器：从分布中采样下一状态
│
└── 控制器（Controller）
    ├── 策略网络：(z, h) -> action
    ├── 探索策略：添加噪声进行探索
    └── 优化器：CMA-ES等进化算法
```

### 完整工作流程

```
1. 感知阶段
   观察 -> VAE编码器 -> 潜在状态z

2. 记忆阶段
   (z_t, a_t, h_t) -> MDN-RNN -> (z_{t+1}, h_{t+1})

3. 决策阶段
   (z, h) -> 控制器 -> 动作a

4. 想象阶段（可选）
   内部模型 -> 想象rollout -> 训练数据
```

## 🚀 快速开始

### 1. 环境要求

- JDK 17 或更高版本
- Maven 3.6+

### 2. 编译模块

```bash
cd /path/to/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn clean compile -pl tinyai-agent-embodied-wm -am
```

### 3. 运行演示程序

```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.wm.WorldModelDemo" \
              -pl tinyai-agent-embodied-wm
```

### 4. 基础使用示例

#### 示例1：创建世界模型

```java
// 1. 创建世界模型配置
WorldModel.WorldModelConfig config = new WorldModel.WorldModelConfig(
    64,    // 观察空间维度
    32,    // 潜在空间维度
    256,   // 隐藏状态维度
    3,     // 动作空间维度
    128,   // VAE隐藏层维度
    5,     // 混合高斯分量数
    false  // 是否确定性策略
);

// 2. 创建世界模型
WorldModel worldModel = new WorldModel(config);
```

#### 示例2：使用智能体交互

```java
// 1. 创建环境和智能体
Environment env = new SimpleDrivingEnvironment();
WorldModelAgent agent = new WorldModelAgent(worldModel, env);

// 2. 运行情景
Episode episode = agent.runEpisode(1000);

System.out.println("情景长度: " + episode.getLength());
System.out.println("总奖励: " + episode.getTotalReward());
```

#### 示例3：想象训练

```java
// 1. 在真实环境中收集初始经验
agent.reset();
for (int i = 0; i < 100; i++) {
    agent.step();
}

// 2. 在想象环境中进行训练
Episode dreamEpisode = agent.trainInDream(500);

System.out.println("想象情景奖励: " + dreamEpisode.getTotalReward());
```

## 📊 核心组件说明

### 1. 数据模型

| 类名 | 说明 | 主要字段 |
|------|------|---------|
| `Observation` | 环境观察 | visualObservation, stateVector |
| `Action` | 智能体动作 | actionVector, actionType |
| `LatentState` | 潜在状态 | z, mu, logVar |
| `HiddenState` | RNN隐藏状态 | h, c (LSTM) |
| `WorldModelState` | 世界模型状态 | latentState, hiddenState |
| `Transition` | 状态转换 | observation, action, reward, nextObservation |
| `Episode` | 情景记录 | transitions, totalReward |

### 2. VAE编码器

**功能**：将高维观察压缩为低维潜在表示

**网络结构**：
```
编码器：
  Input(observationSize) 
  -> Linear(hiddenSize) + ReLU
  -> Linear(hiddenSize) + ReLU
  -> [μ_layer(latentSize), σ²_layer(latentSize)]

重参数化：
  z = μ + σ·ε, ε ~ N(0,1)

解码器：
  Input(latentSize)
  -> Linear(hiddenSize) + ReLU
  -> Linear(hiddenSize) + ReLU
  -> Linear(observationSize)
```

**损失函数**：
```
L_VAE = L_recon + L_KL
L_recon = ||x - x_reconstructed||²
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### 3. MDN-RNN

**功能**：预测潜在状态的时序演化

**网络结构**：
```
输入处理：
  [z_t; a_t] -> Linear(hiddenSize)

GRU单元：
  reset_gate = σ(W_r * [input; h])
  update_gate = σ(W_z * [input; h])
  candidate = tanh(W * [input; r⊙h])
  h_new = (1-z)⊙h + z⊙h_tilde

MDN输出：
  h -> [weights, μ, σ] (混合高斯参数)
```

**损失函数**：
```
L_MDN = -log(Σ π_i · N(z_{t+1}|μ_i, σ_i²))
```

### 4. 控制器

**功能**：基于世界模型状态选择最优动作

**网络结构**：
```
[z; h] 
-> Linear(64) + ReLU
-> Linear(32) + ReLU
-> Linear(actionSize) + Tanh
-> action ∈ [-1, 1]^actionSize
```

**训练方法**：
- CMA-ES（协方差矩阵自适应进化策略）
- 在想象环境中评估适应度
- 无需梯度，适合小规模控制器

## 🎯 技术亮点

### 1. 分离式学习

世界模型采用分离式训练策略：

1. **阶段一：训练VAE**
   - 收集观察数据
   - 训练编码器和解码器
   - 学习压缩的潜在表示

2. **阶段二：训练MDN-RNN**
   - 在潜在空间中收集序列
   - 训练预测下一状态的RNN
   - 学习环境动态模型

3. **阶段三：训练控制器**
   - 在想象环境中进行rollout
   - 使用进化算法优化策略
   - 无需真实环境交互

### 2. 想象训练

智能体可以完全在内部模型中进行训练：

```java
// 想象rollout流程
for (int t = 0; t < dreamSteps; t++) {
    // 1. 控制器选择动作
    action = controller.selectAction(state);
    
    // 2. MDN-RNN预测下一状态
    nextState = mdnRnn.predict(state, action);
    
    // 3. 计算想象奖励
    reward = calculateImaginedReward(state, action, nextState);
    
    // 4. 更新控制器
    updateController(reward);
}
```

**优势**：
- 样本效率高：无需大量真实环境交互
- 训练速度快：内部模型运行速度远超真实环境
- 安全性好：避免在真实环境中的危险探索

### 3. 高斯混合密度网络

使用混合高斯分布建模状态转换的随机性：

```
p(z_{t+1}|z_t, a_t, h_t) = Σ π_i(h_t) · N(μ_i(h_t), σ_i²(h_t))
```

**优势**：
- 可以表示多模态分布
- 捕获环境的随机性
- 比单一高斯更灵活

## 📚 依赖关系

本模块依赖以下TinyAI核心模块：

```xml
<dependencies>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ndarr</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-func</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-nnet</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-ml</artifactId>
    </dependency>
    <dependency>
        <groupId>io.leavesfly.tinyai</groupId>
        <artifactId>tinyai-deeplearning-rl</artifactId>
    </dependency>
</dependencies>
```

## 📖 相关文档

- [**技术架构文档**](doc/技术架构文档.md) - 详细的系统设计文档
- [**TinyAI 主文档**](../README.md) - 项目总体介绍
- [**具身智能模块**](../tinyai-agent-embodied/README.md) - 相关模块参考

## 🎓 学习路径

建议按照以下顺序学习本模块：

1. **理论基础** - 了解世界模型的基本原理和论文
2. **VAE编码** - 学习变分自编码器的实现
3. **MDN-RNN** - 理解混合密度网络和RNN记忆
4. **控制器** - 掌握策略网络和进化算法
5. **想象训练** - 实践在内部模型中训练

## 💡 核心概念

### 世界模型（World Model）

世界模型是智能体对环境的内部表示，包含：

1. **视觉模型（V）**：VAE编码器
   - 压缩高维感知到低维潜在空间
   - 学习环境的视觉特征

2. **记忆模型（M）**：MDN-RNN
   - 预测环境的时序动态
   - 维护历史信息

3. **控制器（C）**：策略网络
   - 基于压缩表示做决策
   - 可在想象中训练

### 端到端学习

直接从原始观察学习到动作映射：

```
Raw Observation -> VAE -> Latent z -> Controller -> Action
```

**优势**：
- 无需手工特征工程
- 端到端优化整个流程
- 更好的泛化能力

## 🔬 技术参数

### 默认配置

```java
observationSize = 64     // 观察向量维度
latentSize = 32         // 潜在空间维度
hiddenSize = 256        // RNN隐藏状态维度
actionSize = 3          // 动作空间维度
vaeHiddenSize = 128     // VAE隐藏层维度
numMixtures = 5         // 混合高斯分量数
deterministic = false   // 随机策略
```

### 训练超参数

```java
// VAE训练
vaeLearningRate = 0.001
vaeEpochs = 100
vaeBatchSize = 32

// MDN-RNN训练
rnnLearningRate = 0.001
rnnEpochs = 50
rnnSequenceLength = 32

// 控制器训练（CMA-ES）
populationSize = 16
sigma = 0.1
generations = 100
```

## 🧪 代码示例

### 完整训练流程

```java
// 1. 创建环境和模型
Environment env = new SimpleDrivingEnvironment();
WorldModel worldModel = new WorldModel(WorldModel.WorldModelConfig.createDefault());
WorldModelAgent agent = new WorldModelAgent(worldModel, env);

// 2. 收集训练数据（真实环境）
List<Episode> realEpisodes = new ArrayList<>();
for (int i = 0; i < 100; i++) {
    Episode episode = agent.runEpisode(1000);
    realEpisodes.add(episode);
}

// 3. 训练VAE（离线）
trainVAE(worldModel.getVaeEncoder(), realEpisodes);

// 4. 训练MDN-RNN（离线）
trainMDNRNN(worldModel.getMdnRnn(), realEpisodes);

// 5. 训练控制器（想象环境）
for (int i = 0; i < 1000; i++) {
    Episode dreamEpisode = agent.trainInDream(100);
    updateController(worldModel.getController(), dreamEpisode);
}

// 6. 评估性能
double avgReward = agent.evaluate(10);
System.out.println("平均奖励: " + avgReward);
```

## 📊 模块统计

| 类别 | 数量 | 说明 |
|-----|------|------|
| Java 类文件 | 15+ | 包括所有核心组件 |
| 数据模型 | 8个 | 观察、动作、状态等 |
| 核心组件 | 3个 | VAE、MDN-RNN、Controller |
| 环境实现 | 1个 | 简单驾驶环境 |
| 演示程序 | 1个 | 完整使用示例 |

## 🛠️ 技术栈

| 项目 | 版本/配置 | 说明 |
|-----|----------|------|
| Java | JDK 17+ | 核心语言 |
| Maven | 3.6+ | 构建工具 |
| TinyAI NdArray | 1.0.0 | 多维数组库 |
| TinyAI AutoGrad | 1.0.0 | 自动微分 |
| TinyAI NeuralNet | 1.0.0 | 神经网络 |

## ❓ 常见问题

### Q1: 世界模型与传统强化学习有什么区别？

**A**: 世界模型学习环境的内部表示，可以在想象中训练，大大提高样本效率。传统RL需要大量真实环境交互。

### Q2: 为什么使用混合密度网络？

**A**: 环境转换往往是随机的和多模态的，单一高斯无法很好建模。MDN可以表示复杂的概率分布。

### Q3: 如何调整潜在空间维度？

**A**: 通过配置参数调整：
```java
config.setLatentSize(64);  // 增加到64维
```

### Q4: 控制器可以使用其他优化方法吗？

**A**: 可以。除了CMA-ES，还可以使用梯度下降、PPO等方法训练控制器。

## 🔗 参考资料

- [World Models论文](https://worldmodels.github.io/) - David Ha & Jürgen Schmidhuber
- [VAE原理](https://arxiv.org/abs/1312.6114) - Kingma & Welling
- [MDN原理](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) - Bishop, 1994

## 📝 更新日志

### v1.0.0 (2025-10-18)
- ✅ 实现完整的世界模型架构
- ✅ VAE编码器、MDN-RNN、控制器
- ✅ 想象训练功能
- ✅ 简单驾驶环境
- ✅ 演示程序和文档

---

**TinyAI 世界模型** - 让AI在想象中学习! 💭🧠
