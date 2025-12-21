package io.leavesfly.tinyai.banana.block;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.banana.encoder.ImageEncoder;
import io.leavesfly.tinyai.banana.encoder.TextEncoder;
import io.leavesfly.tinyai.banana.fusion.MultiModalFusion;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * Banana模型主体块
 * 
 * 整合所有Banana组件,构建完整的多模态架构:
 * 1. 文本编码器 - 处理文本输入
 * 2. 图像编码器(ViT) - 处理图像输入
 * 3. 多模态融合层 - 跨模态注意力机制
 * 4. 输出投影层 - 生成最终输出
 * 
 * 数据流:
 * text_tokens → TextEncoder → multimodal_fusion ↘
 * image_pixels → ImageEncoder → multimodal_fusion → output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class BananaBlock extends Module {
    
    private final BananaConfig config;
    
    // 核心组件
    private TextEncoder textEncoder;        // 文本编码器
    private ImageEncoder imageEncoder;      // 图像编码器
    private MultiModalFusion fusionLayer;   // 多模态融合层
    
    // 基础组件
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public BananaBlock(String name, BananaConfig config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件
     */
    private void initializeComponents() {
        // 初始化文本编码器
        textEncoder = new TextEncoder(name + "_text_encoder", config);
        registerModule("text_encoder", textEncoder);
        
        // 初始化图像编码器
        imageEncoder = new ImageEncoder(name + "_image_encoder", config);
        registerModule("image_encoder", imageEncoder);
        
        // 初始化多模态融合层
        if (config.isEnableCrossModalAttention()) {
            fusionLayer = new MultiModalFusion(name + "_fusion", config);
            registerModule("fusion", fusionLayer);
        }
        
        // 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln",
            config.getHiddenSize(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 初始化输出投影层
        // 根据任务类型,可能输出到不同的空间
        // 文本生成: vocab_size
        // 图像生成: image_vocab_size
        outputProjection = new Linear(
            name + "_output_proj",
            config.getHiddenSize(),
            config.getVocabSize(),  // 默认文本词汇表
            false
        );
        registerModule("output_proj", outputProjection);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为文本token IDs或图像数据
     * @return 输出Variable
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        
        // 简化实现:直接通过LayerNorm和输出投影
        // TODO: 后续实现完整的编码器和融合逻辑
        Variable x = finalLayerNorm.forward(input);
        Variable output = outputProjection.forward(x);
        
        return output;
    }
    
    /**
     * 文本编码前向传播
     * 
     * @param textTokenIds 文本token IDs [batch, seq_len]
     * @return 文本特征 [batch, seq_len, hidden_size]
     */
    public Variable forwardText(Variable textTokenIds) {
        return textEncoder.forward(textTokenIds);
    }
    
    /**
     * 图像编码前向传播
     * 
     * @param imagePixels 图像像素 [batch, channels, height, width]
     * @return 图像特征 [batch, num_patches, hidden_size]
     */
    public Variable forwardImage(Variable imagePixels) {
        return imageEncoder.forward(imagePixels);
    }
    
    /**
     * 多模态融合前向传播
     * 
     * @param textFeatures 文本特征 [batch, text_len, hidden_size]
     * @param imageFeatures 图像特征 [batch, num_patches, hidden_size]
     * @param taskType 任务类型
     * @return 融合后的特征 [batch, text_len, hidden_size] (不做输出投影)
     */
    public Variable forwardMultiModal(Variable textFeatures, 
                                     Variable imageFeatures, 
                                     TaskType taskType) {
        if (fusionLayer != null) {
            // 跨模态融合 - 返回融合后的文本特征
            Variable fusedTextFeatures = fusionLayer.forward(textFeatures, imageFeatures);
            
            // 最终LayerNorm (不做输出投影,保持在特征空间)
            return finalLayerNorm.forward(fusedTextFeatures);
        } else {
            // 如果未启用跨模态，直接返回归一化后的文本特征
            return finalLayerNorm.forward(textFeatures);
        }
    }
    
    /**
     * 多模态融合前向传播(带输出投影)
     * 
     * 用于生成任务,将特征投影到词汇表空间
     * 
     * @param textFeatures 文本特征 [batch, text_len, hidden_size]
     * @param imageFeatures 图像特征 [batch, num_patches, hidden_size]
     * @param taskType 任务类型
     * @return 投影后的输出 [batch, text_len, vocab_size]
     */
    public Variable forwardMultiModalWithProjection(Variable textFeatures, 
                                                    Variable imageFeatures, 
                                                    TaskType taskType) {
        // 先进行融合
        Variable fusedFeatures = forwardMultiModal(textFeatures, imageFeatures, taskType);
        
        // 再进行输出投影
        return outputProjection.forward(fusedFeatures);
    }
    
    /**
     * 带详细信息的前向传播
     * 
     * @param textTokenIds 文本token IDs
     * @param imagePixels 图像像素(可选)
     * @param taskType 任务类型
     * @return 详细的前向结果
     */
    public DetailedForwardResult forwardWithDetails(Variable textTokenIds,
                                                    Variable imagePixels,
                                                    TaskType taskType) {
        // TODO: 实现完整的多模态流程
        throw new UnsupportedOperationException("详细前向传播尚未实现");
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable input) {
        if (input == null) {
            throw new IllegalArgumentException("输入Variable不能为null");
        }
        
        int[] shape = input.getValue().getShape().getShapeDims();
        if (shape.length < 2) {
            throw new IllegalArgumentException(
                "输入shape至少需要2维 [batch, seq_len], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
    }
    
    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(80));
        System.out.println("Banana模型架构");
        System.out.println("=".repeat(80));
        System.out.println("配置: " + config);
        System.out.println("-".repeat(80));
        System.out.println("组件状态:");
        System.out.println("  文本编码器: ✓ " + textEncoder);
        System.out.println("  图像编码器: ✓ " + imageEncoder);
        System.out.println("  多模态融合: " + 
            (fusionLayer != null ? "✓ " + fusionLayer : "未启用"));
        System.out.println("  最终LayerNorm: ✓");
        System.out.println("  输出投影层: ✓");
        System.out.println("=".repeat(80));
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public TextEncoder getTextEncoder() {
        return textEncoder;
    }
    
    public ImageEncoder getImageEncoder() {
        return imageEncoder;
    }
    
    public MultiModalFusion getFusionLayer() {
        return fusionLayer;
    }
    
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    public Linear getOutputProjection() {
        return outputProjection;
    }
    
    // ==================== 内部结果类 ====================
    
    /**
     * 详细前向传播结果
     */
    public static class DetailedForwardResult {
        public final Variable output;
        public final Variable textFeatures;
        public final Variable imageFeatures;
        public final Variable fusedFeatures;
        public final TaskType taskType;
        
        public DetailedForwardResult(Variable output,
                                    Variable textFeatures,
                                    Variable imageFeatures,
                                    Variable fusedFeatures,
                                    TaskType taskType) {
            this.output = output;
            this.textFeatures = textFeatures;
            this.imageFeatures = imageFeatures;
            this.fusedFeatures = fusedFeatures;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format(
                "DetailedForwardResult{taskType=%s, hasTextFeatures=%b, hasImageFeatures=%b}",
                taskType, 
                textFeatures != null, 
                imageFeatures != null
            );
        }
    }
}
