# TinyAI Banana - API参考文档

## 📋 文档信息

- **模块**: tinyai-model-banana
- **版本**: v1.0  
- **最后更新**: 2025-12-21

---

## 📑 目录

1. [配置管理API](#配置管理api)
2. [编码器API](#编码器api)
3. [融合层API](#融合层api)
4. [模型接口API](#模型接口api)
5. [使用示例](#使用示例)

---

## 一、配置管理API

### 1.1 BananaConfig

Banana模型的配置类,支持预设配置和完全自定义。

#### 构造函数

```java
// 默认构造(创建Tiny配置)
public BananaConfig()

// 完全自定义配置
public BananaConfig(
    int vocabSize,
    int maxTextLength,
    int hiddenSize,
    int numLayers,
    int numHeads,
    int ffnHiddenSize,
    int imageSize,
    int patchSize
)
```

#### 预设配置工厂方法

```java
// 创建Tiny配置 (60M参数)
public static BananaConfig createTinyConfig()

// 创建Small配置 (167M参数)
public static BananaConfig createSmallConfig()

// 创建Base配置 (386M参数)
public static BananaConfig createBaseConfig()
```

**使用示例**:

```java
// 方式1: 使用预设
BananaConfig config = BananaConfig.createTinyConfig();

// 方式2: 自定义配置
BananaConfig config = new BananaConfig();
config.setHiddenSize(768);
config.setNumLayers(12);
config.setImageSize(384);
config.updateNumPatches();
```

#### 核心配置项

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getVocabSize()` | int | 词汇表大小 |
| `getMaxTextLength()` | int | 最大文本序列长度 |
| `getHiddenSize()` | int | 隐藏层维度 |
| `getNumLayers()` | int | Transformer层数 |
| `getNumHeads()` | int | 注意力头数 |
| `getFfnHiddenSize()` | int | FFN隐藏层维度 |
| `getImageSize()` | int | 图像尺寸(宽高) |
| `getPatchSize()` | int | Patch尺寸 |
| `getNumPatches()` | int | Patch数量 |
| `getImageChannels()` | int | 图像通道数(3 for RGB) |
| `getNumEncoderLayers()` | int | 图像编码器层数 |
| `isEnableCrossModalAttention()` | boolean | 是否启用跨模态注意力 |

#### 配置验证

```java
// 验证配置有效性
public void validate() throws IllegalArgumentException

// 更新计算字段
public void updateNumPatches()  // 根据imageSize和patchSize计算

// 参数量估算
public long estimateParameters()

// 格式化输出参数量
public String formatParameters()  // 如 "60.82M"
```

#### 配置摘要

```java
// 获取配置摘要
public String getConfigSummary()

// toString输出详细配置
@Override
public String toString()
```

---

## 二、编码器API

### 2.1 TextEncoder

文本编码器,基于Transformer架构处理文本输入。

#### 构造函数

```java
public TextEncoder(String name, BananaConfig config)
```

**参数**:
- `name`: 编码器名称
- `config`: Banana配置对象

#### 前向传播

```java
public Variable forward(Variable... inputs)
```

**输入**:
- `inputs[0]`: Token IDs, 形状 `[batch, seq_len]`

**输出**:
- 文本特征, 形状 `[batch, seq_len, hidden_size]`

**使用示例**:

```java
BananaConfig config = BananaConfig.createTinyConfig();
TextEncoder encoder = new TextEncoder("text_enc", config);

// 创建Token IDs (假设vocab_size=32000)
NdArray tokenIds = NdArray.of(Shape.of(2, 10));  // 2个样本,序列长度10
Variable input = new Variable(tokenIds);

// 编码
Variable textFeatures = encoder.forward(input);
// 输出shape: [2, 10, 512]
```

### 2.2 ImageEncoder

图像编码器,基于Vision Transformer处理图像输入。

#### 构造函数

```java
public ImageEncoder(String name, BananaConfig config)
```

#### 前向传播

```java
public Variable forward(Variable... inputs)
```

**输入**:
- `inputs[0]`: 图像像素, 形状 `[batch, channels, height, width]`
  - channels: 3 (RGB)
  - height/width: 必须等于config.imageSize

**输出**:
- 图像特征, 形状 `[batch, num_patches, hidden_size]`

**使用示例**:

```java
BananaConfig config = BananaConfig.createTinyConfig();
ImageEncoder encoder = new ImageEncoder("image_enc", config);

// 创建图像数据 [batch=2, channels=3, height=256, width=256]
NdArray imageData = NdArray.of(Shape.of(2, 3, 256, 256));
Variable imageInput = new Variable(imageData);

// 编码
Variable imageFeatures = encoder.forward(imageInput);
// 输出shape: [2, 256, 512]  (256个patches,每个512维)
```

### 2.3 PatchEmbedding

图像切片嵌入层,将图像分割成patches。

#### 构造函数

```java
public PatchEmbedding(
    String name,
    int imageSize,
    int patchSize,
    int imageChannels,
    int hiddenSize
)
```

**参数约束**:
- `imageSize`必须能被`patchSize`整除

#### 前向传播

```java
public Variable forward(Variable... inputs)
```

**输入**:
- `inputs[0]`: 图像 `[batch, channels, height, width]`

**输出**:
- Patch序列 `[batch, num_patches, hidden_size]`

**技术细节**:
```java
// num_patches = (imageSize / patchSize)^2
// 例如: imageSize=256, patchSize=16
// → num_patches = (256/16)^2 = 256
```

### 2.4 Position2D

2D位置编码,为图像patches添加空间位置信息。

#### 构造函数

```java
public Position2D(
    String name,
    int numPatches,
    int hiddenSize
)
```

#### 前向传播

```java
public Variable forward(Variable... inputs)
```

**输入**:
- `inputs[0]`: Patch序列 (可选,位置编码独立于输入)

**输出**:
- 位置编码 `[1, num_patches, hidden_size]`
- 第一维为1,可广播到任意batch_size

#### 位置查询

```java
// 根据patch索引获取位置编码
public Variable getPositionAt(int patchIndex)

// 根据2D坐标获取位置编码
public Variable getPositionAt2D(int row, int col, int numPatchesPerRow)
```

---

## 三、融合层API

### 3.1 CrossModalAttention

跨模态注意力层,实现两个模态之间的注意力交互。

#### 构造函数

```java
public CrossModalAttention(
    String name,
    int hiddenSize,
    int numHeads,
    float dropout
)
```

**参数**:
- `hiddenSize`: 隐藏层维度
- `numHeads`: 注意力头数(必须能整除hiddenSize)
- `dropout`: Dropout比率

#### 前向传播

```java
public Variable forward(Variable... inputs)
```

**输入**:
- `inputs[0]`: Query特征 (如文本) `[batch, query_len, hidden_size]`
- `inputs[1]`: Key/Value特征 (如图像) `[batch, kv_len, hidden_size]`

**输出**:
- 融合后的Query特征 `[batch, query_len, hidden_size]`

**使用示例**:

```java
CrossModalAttention crossAttn = new CrossModalAttention(
    "text2image",
    512,   // hiddenSize
    8,     // numHeads
    0.1f   // dropout
);

// 文本特征: [2, 10, 512]
// 图像特征: [2, 256, 512]
Variable fusedText = crossAttn.forward(textFeatures, imageFeatures);
// 输出: [2, 10, 512]  文本关注了图像信息
```

### 3.2 MultiModalFusion

多模态融合模块,实现文本-图像的双向注意力融合。

#### 构造函数

```java
public MultiModalFusion(String name, BananaConfig config)
```

#### 前向传播

```java
// 单向融合(仅返回文本融合结果)
public Variable forward(Variable... inputs)

// 双向融合(同时返回文本和图像融合结果)
public Variable[] forwardBoth(Variable textFeatures, Variable imageFeatures)
```

**输入**:
- `textFeatures`: 文本特征 `[batch, text_len, hidden_size]`
- `imageFeatures`: 图像特征 `[batch, num_patches, hidden_size]`

**输出**:
- `forward`: 融合后的文本特征
- `forwardBoth`: [融合文本特征, 融合图像特征]

**使用示例**:

```java
BananaConfig config = BananaConfig.createTinyConfig();
MultiModalFusion fusion = new MultiModalFusion("fusion", config);

// 双向融合
Variable[] fused = fusion.forwardBoth(textFeatures, imageFeatures);
Variable fusedText = fused[0];   // 文本融合了图像信息
Variable fusedImage = fused[1];  // 图像融合了文本信息
```

---

## 四、模型接口API

### 4.1 BananaBlock

Banana模型的主体模块,整合所有编码器和融合层。

#### 构造函数

```java
public BananaBlock(String name, BananaConfig config)
```

#### 前向传播方法

```java
// 仅文本编码
public Variable forwardText(Variable textTokenIds)

// 仅图像编码
public Variable forwardImage(Variable imagePixels)

// 多模态融合
public Variable forwardMultiModal(
    Variable textFeatures,
    Variable imageFeatures,
    TaskType taskType
)
```

**参数**:
- `textTokenIds`: Token IDs `[batch, text_len]`
- `imagePixels`: 图像像素 `[batch, 3, H, W]`
- `textFeatures`: 文本特征 `[batch, text_len, hidden_size]`
- `imageFeatures`: 图像特征 `[batch, num_patches, hidden_size]`
- `taskType`: 任务类型(用于未来扩展)

#### 模型信息

```java
// 打印模型详细信息
public void printModelInfo()

// 获取配置
public BananaConfig getConfig()
```

### 4.2 BananaModel

Banana模型的接口类,继承自`Model`基类。

#### 创建模型

```java
// 使用预设配置创建
public static BananaModel create(String name, String preset)

// 使用自定义配置创建
public BananaModel(String name, BananaConfig config)
```

**预设类型**:
- `"tiny"`: Tiny配置 (60M参数)
- `"small"`: Small配置 (167M参数)
- `"base"`: Base配置 (386M参数)

**使用示例**:

```java
// 方式1: 预设配置
BananaModel model = BananaModel.create("banana_tiny", "tiny");

// 方式2: 自定义配置
BananaConfig config = new BananaConfig();
config.setHiddenSize(768);
BananaModel model = new BananaModel("banana_custom", config);
```

#### 编码方法

```java
// 文本编码
public Variable encodeText(Variable textTokenIds)

// 图像编码
public Variable encodeImage(Variable imagePixels)

// 文本生成图像(待实现)
public Variable generateImage(Variable textTokenIds)
```

#### 模型信息

```java
// 获取模型名称
public String getName()

// 获取配置
public BananaConfig getConfig()

// 获取配置摘要
public String getConfigSummary()

// toString输出
@Override
public String toString()  // 返回 "BananaModel{...}"
```

---

## 五、使用示例

### 5.1 完整工作流程

```java
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

public class BananaExample {
    public static void main(String[] args) {
        // 1. 创建模型
        BananaModel model = BananaModel.create("banana", "tiny");
        
        // 2. 准备输入数据
        // 文本输入: [batch=2, seq_len=10]
        NdArray textData = NdArray.of(Shape.of(2, 10));
        Variable textInput = new Variable(textData);
        
        // 图像输入: [batch=2, channels=3, height=256, width=256]
        NdArray imageData = NdArray.of(Shape.of(2, 3, 256, 256));
        Variable imageInput = new Variable(imageData);
        
        // 3. 文本编码
        Variable textFeatures = model.encodeText(textInput);
        System.out.println("文本特征: " + textFeatures.getValue().getShape());
        // 输出: [2, 10, 512]
        
        // 4. 图像编码
        Variable imageFeatures = model.encodeImage(imageInput);
        System.out.println("图像特征: " + imageFeatures.getValue().getShape());
        // 输出: [2, 256, 512]
        
        // 5. 多模态融合(通过Block直接调用)
        // Variable fusedOutput = model.getBananaBlock()
        //     .forwardMultiModal(textFeatures, imageFeatures, TaskType.TEXT_TO_IMAGE);
    }
}
```

### 5.2 配置自定义示例

```java
// 创建自定义配置
BananaConfig config = new BananaConfig();

// 基础配置
config.setVocabSize(50000);
config.setMaxTextLength(1024);
config.setHiddenSize(768);
config.setNumLayers(12);
config.setNumHeads(12);
config.setFfnHiddenSize(3072);

// 图像配置
config.setImageSize(384);
config.setPatchSize(16);
config.updateNumPatches();  // 计算: (384/16)^2 = 576

// 多模态配置
config.setEnableCrossModalAttention(true);
config.setNumEncoderLayers(9);

// 验证配置
try {
    config.validate();
    System.out.println("配置验证通过");
} catch (IllegalArgumentException e) {
    System.err.println("配置错误: " + e.getMessage());
}

// 创建模型
BananaModel model = new BananaModel("custom_model", config);
System.out.println("参数量: " + config.formatParameters());
```

### 5.3 批量处理示例

```java
// 批量处理图像
public void batchProcessImages(BananaModel model, List<NdArray> images) {
    int batchSize = images.size();
    
    // 堆叠成batch
    NdArray batchImages = stackImages(images);  // [batch, 3, 256, 256]
    Variable input = new Variable(batchImages);
    
    // 批量编码
    Variable features = model.encodeImage(input);
    
    // 处理每个样本的特征
    for (int i = 0; i < batchSize; i++) {
        // 提取单个样本特征: features[i, :, :]
        // ... 后续处理
    }
}
```

### 5.4 性能监控示例

```java
// 性能测试
public void benchmarkModel() {
    BananaModel model = BananaModel.create("benchmark", "tiny");
    
    // 准备测试数据
    Variable imageInput = createRandomImage(2, 256);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        model.encodeImage(imageInput);
    }
    
    // 正式测试
    long startTime = System.currentTimeMillis();
    int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        Variable output = model.encodeImage(imageInput);
    }
    
    long endTime = System.currentTimeMillis();
    double avgTime = (endTime - startTime) / (double) iterations;
    
    System.out.println("平均编码时间: " + avgTime + "ms");
}
```

---

## 附录

### A. 任务类型枚举

```java
public enum TaskType {
    TEXT_TO_IMAGE("文本生成图像"),
    IMAGE_TO_TEXT("图像生成描述"),
    IMAGE_EDITING("图像编辑"),
    MULTIMODAL_UNDERSTANDING("多模态理解"),
    ZERO_SHOT_CLASSIFICATION("零样本分类");
    
    private final String description;
    
    TaskType(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
}
```

### B. 常见错误处理

| 错误类型 | 原因 | 解决方法 |
|---------|------|---------|
| `IllegalArgumentException` | hiddenSize不能被numHeads整除 | 调整hiddenSize或numHeads |
| `IllegalArgumentException` | imageSize不能被patchSize整除 | 调整imageSize或patchSize |
| `IllegalArgumentException` | 输入图像尺寸不匹配 | 确保图像尺寸等于config.imageSize |
| `IllegalArgumentException` | 输入通道数错误 | 确保图像为RGB(3通道) |

### C. 性能优化建议

1. **批处理**: 尽量使用较大的batch_size提升吞吐量
2. **模型预热**: 首次运行较慢,预热后性能提升2-3倍
3. **配置选择**: 根据硬件资源选择合适的模型规模
4. **梯度检查点**: 大模型训练时可启用梯度检查点减少内存

---

**API文档完成**: 本文档提供了tinyai-model-banana模块的完整API参考,包括所有公开接口、使用示例和最佳实践。

**最后更新**: 2025-12-21  
**文档版本**: v1.0
