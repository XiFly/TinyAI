package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DeepSeek-R1模型类
 * 
 * DeepSeek-R1是一个具备深度推理和自我反思能力的大语言模型，
 * 通过多步推理和反思机制实现复杂任务的可解释性处理。
 * 
 * 主要特性：
 * 1. 多步推理 - 支持最多7步迭代推理过程
 * 2. 自我反思 - 从5个维度评估推理质量
 * 3. 置信度评估 - 动态评估每步推理的可信度
 * 4. Pre-LayerNorm架构 - 提升训练稳定性
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Model extends Model {
    
    private final DeepSeekR1Config config;
    private final DeepSeekR1Block r1Block;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param config R1配置对象
     */
    public DeepSeekR1Model(String name, DeepSeekR1Config config) {
        super(name, new DeepSeekR1Block(name + "_main", config));
        this.config = config;
        this.r1Block = (DeepSeekR1Block) getModule();
        setDescription(buildDescription());
    }
    
    /**
     * 构建模型描述信息
     */
    private String buildDescription() {
        return String.format(
            "DeepSeek-R1语言模型 | 参数量: %s | 层数: %d | 维度: %d | 注意力头: %d | " +
            "推理步骤: %d | 架构: Pre-LayerNorm",
            formatParamCount(config.estimateParameterCount()),
            config.getNLayer(),
            config.getNEmbd(),
            config.getNHead(),
            config.getMaxReasoningSteps()
        );
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== 工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-R1模型
     */
    public static DeepSeekR1Model createStandardModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createStandardConfig());
    }
    
    /**
     * 创建微型DeepSeek-R1模型（用于快速测试）
     */
    public static DeepSeekR1Model createTinyModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createTinyConfig());
    }
    
    /**
     * 创建小型DeepSeek-R1模型（用于学习和实验）
     */
    public static DeepSeekR1Model createSmallModel(String name) {
        return new DeepSeekR1Model(name, DeepSeekR1Config.createSmallConfig());
    }
    
    // ==================== 推理方法 ====================
    
    /**
     * 标准预测方法
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    /**
     * 带详细信息的推理
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 详细推理结果，包含推理步骤和反思评估
     */
    public DeepSeekR1Block.DetailedForwardResult predictWithDetails(Variable tokenIds) {
        return r1Block.forwardWithDetails(tokenIds);
    }
    
    /**
     * 执行多步推理（获取推理结果）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return 推理结果对象
     */
    public ReasoningOutput performReasoning(Variable tokenIds) {
        DeepSeekR1Block.DetailedForwardResult result = r1Block.forwardWithDetails(tokenIds);
        return new ReasoningOutput(
            result.logits,
            result.reasoningResult.numSteps,
            result.reasoningResult.averageConfidence,
            result.reflectionResult.qualityScore
        );
    }
    
    /**
     * 生成序列（贪婪解码）
     * 
     * @param promptIds 提示词token ID序列 [batch_size, prompt_len]
     * @param maxNewTokens 最大生成token数量
     * @return 生成的完整序列 [batch_size, prompt_len + maxNewTokens]
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);
        
        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];
        
        // 复制提示词
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }
        
        // 自回归生成
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;
            float[][] currentInput = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, currentLen);
            }
            
            // 预测下一个token
            Variable logits = predict(new Variable(NdArray.of(currentInput)));
            NdArray logitsArray = logits.getValue();
            
            // 贪婪选择（选择概率最大的token）
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
            }
        }
        
        return NdArray.of(generatedSeq);
    }
    
    /**
     * 查找最大值的索引（argmax）
     */
    private int argmax(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 0;
        float maxVal = logits.get(batchIdx, seqIdx, 0);
        
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    // ==================== 模型信息 ====================
    
    /**
     * 打印模型详细信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 模型详细信息");
        System.out.println("=".repeat(80));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(80));
        System.out.println(config);
        System.out.println("-".repeat(80));
        if (r1Block != null) {
            r1Block.printArchitecture();
        }
        System.out.println("=".repeat(80));
    }
    
    /**
     * 获取配置摘要
     */
    public String getConfigSummary() {
        return String.format(
            "DeepSeek-R1配置摘要:\n" +
            "  - 词汇表大小: %,d\n" +
            "  - 嵌入维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - 前馈网络维度: %d\n" +
            "  - 最大序列长度: %d\n" +
            "  - 最大推理步骤: %d\n" +
            "  - 推理隐藏维度: %d\n" +
            "  - 反思隐藏维度: %d\n" +
            "  - 质量评分维度: %d\n" +
            "  - 置信度阈值: %.2f\n" +
            "  - 架构: Pre-LayerNorm\n" +
            "  - 估算参数量: %s",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            config.getMaxReasoningSteps(),
            config.getReasoningHiddenDim(),
            config.getReflectionHiddenDim(),
            config.getQualityScoreDim(),
            config.getConfidenceThreshold(),
            formatParamCount(config.estimateParameterCount())
        );
    }
    
    // ==================== Getter方法 ====================
    
    public DeepSeekR1Config getConfig() {
        return config;
    }
    
    public DeepSeekR1Block getR1Block() {
        return r1Block;
    }
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekR1Model{name='%s', params=%s, nLayer=%d, nEmbd=%d, reasoningSteps=%d}",
            getName(), 
            formatParamCount(config.estimateParameterCount()), 
            config.getNLayer(), 
            config.getNEmbd(),
            config.getMaxReasoningSteps()
        );
    }
    
    // ==================== 内部类 ====================
    
    /**
     * 推理输出结果类
     */
    public static class ReasoningOutput {
        /** 最终logits输出 */
        public final Variable logits;
        /** 推理步骤数 */
        public final int numSteps;
        /** 平均置信度 */
        public final double averageConfidence;
        /** 质量评分 */
        public final DeepSeekR1ReflectionBlock.QualityScore qualityScore;
        
        public ReasoningOutput(Variable logits, int numSteps, 
                              double averageConfidence,
                              DeepSeekR1ReflectionBlock.QualityScore qualityScore) {
            this.logits = logits;
            this.numSteps = numSteps;
            this.averageConfidence = averageConfidence;
            this.qualityScore = qualityScore;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReasoningOutput{\n" +
                "  推理步骤数: %d\n" +
                "  平均置信度: %.4f\n" +
                "  %s\n" +
                "}",
                numSteps, averageConfidence, qualityScore
            );
        }
    }
}
