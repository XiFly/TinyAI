package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-R1主体块（DeepSeekR1Block）
 * 
 * 整合所有DeepSeek-R1组件，构建完整的模型架构：
 * 1. Token嵌入层 - 将token ID转换为向量表示
 * 2. Transformer层堆叠 - 进行序列建模
 * 3. 推理模块 - 执行多步推理
 * 4. 反思模块 - 进行自我评估
 * 5. 输出投影层 - 生成最终logits
 * 
 * 数据流：
 * token_ids → embedding → transformer_layers → reasoning → reflection → output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Block extends Module {
    
    private final DeepSeekR1Config config;
    
    // 核心组件
    private DeepSeekR1TokenEmbedding tokenEmbedding;
    private List<DeepSeekR1TransformerBlock> transformerBlocks;
    private DeepSeekR1ReasoningBlock reasoningBlock;
    private DeepSeekR1ReflectionBlock reflectionBlock;
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1Block(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层
        tokenEmbedding = new DeepSeekR1TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        // 2. 初始化Transformer层堆叠
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            DeepSeekR1TransformerBlock block = new DeepSeekR1TransformerBlock(
                name + "_transformer_" + i, config);
            transformerBlocks.add(block);
            registerModule("transformer_" + i, block);
        }
        
        // 3. 初始化推理模块
        reasoningBlock = new DeepSeekR1ReasoningBlock(name + "_reasoning", config);
        registerModule("reasoning", reasoningBlock);
        
        // 4. 初始化反思模块
        reflectionBlock = new DeepSeekR1ReflectionBlock(name + "_reflection", config);
        registerModule("reflection", reflectionBlock);
        
        // 5. 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln",
            config.getNEmbd(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 6. 初始化输出投影层
        outputProjection = new Linear(
            name + "_output_proj",
            config.getNEmbd(),
            config.getVocabSize(),
            false  // 通常不使用偏置
        );
        registerModule("output_proj", outputProjection);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs 输入变量，inputs[0]为token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠
        for (DeepSeekR1TransformerBlock block : transformerBlocks) {
            x = block.forward(x);
        }
        
        // 3. 推理模块（可选地获取推理结果）
        Variable reasoningOutput = reasoningBlock.forward(x);
        
        // 4. 反思模块（可选地获取反思结果）
        Variable reflectionOutput = reflectionBlock.forward(reasoningOutput);
        
        // 5. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(reflectionOutput);
        
        // 6. 输出投影
        Variable logits = outputProjection.forward(normalized);
        
        return logits;
    }
    
    /**
     * 带详细输出的前向传播（包含推理和反思结果）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable tokenIds) {
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠
        for (DeepSeekR1TransformerBlock block : transformerBlocks) {
            x = block.forward(x);
        }
        
        // 3. 推理模块（获取详细结果）
        DeepSeekR1ReasoningBlock.ReasoningResult reasoningResult = 
            reasoningBlock.performMultiStepReasoning(x);
        
        // 4. 反思模块（获取详细结果）
        DeepSeekR1ReflectionBlock.ReflectionResult reflectionResult = 
            reflectionBlock.performReflection(reasoningResult.reasoningOutput);
        
        // 5. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(reflectionResult.suggestionOutput);
        
        // 6. 输出投影
        Variable logits = outputProjection.forward(normalized);
        
        return new DetailedForwardResult(logits, reasoningResult, reflectionResult);
    }
    
    /**
     * 验证输入的有效性
     * 
     * @param tokenIds token ID变量
     */
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape())
            );
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    /**
     * 估算参数数量
     */
    public long getParameterCount() {
        return config.estimateParameterCount();
    }
    
    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 主体块架构");
        System.out.println("=".repeat(70));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(70));
        System.out.printf("Token嵌入层: %s\n", tokenEmbedding.getClass().getSimpleName());
        System.out.printf("Transformer块数量: %d\n", transformerBlocks.size());
        System.out.printf("推理模块: %s (最大%d步)\n", 
            reasoningBlock.getClass().getSimpleName(), config.getMaxReasoningSteps());
        System.out.printf("反思模块: %s (%d维质量评分)\n", 
            reflectionBlock.getClass().getSimpleName(), config.getQualityScoreDim());
        System.out.printf("架构模式: Pre-LayerNorm\n");
        System.out.printf("估算参数数量: %s\n", formatParamCount(getParameterCount()));
        System.out.println("=".repeat(70));
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    /**
     * 详细前向传播结果类
     */
    public static class DetailedForwardResult {
        /** 最终logits输出 */
        public final Variable logits;
        /** 推理结果 */
        public final DeepSeekR1ReasoningBlock.ReasoningResult reasoningResult;
        /** 反思结果 */
        public final DeepSeekR1ReflectionBlock.ReflectionResult reflectionResult;
        
        public DetailedForwardResult(Variable logits,
                                    DeepSeekR1ReasoningBlock.ReasoningResult reasoningResult,
                                    DeepSeekR1ReflectionBlock.ReflectionResult reflectionResult) {
            this.logits = logits;
            this.reasoningResult = reasoningResult;
            this.reflectionResult = reflectionResult;
        }
        
        @Override
        public String toString() {
            return String.format(
                "DetailedForwardResult{\n" +
                "  %s\n" +
                "  %s\n" +
                "}",
                reasoningResult, reflectionResult
            );
        }
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
    
    /**
     * 获取Transformer块列表
     */
    public List<DeepSeekR1TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
}
