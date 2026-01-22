package io.leavesfly.tinyai.deepseek.r1;

import java.io.Serializable;

/**
 * DeepSeek-R1模型配置类
 * 
 * DeepSeek-R1是一个具备深度推理和自我反思能力的大语言模型，
 * 通过多步推理和反思机制实现复杂任务的可解释性处理。
 * 
 * 核心特点：
 * 1. 多步推理能力 - 支持7步迭代推理过程
 * 2. 自我反思机制 - 推理质量评估和改进建议生成
 * 3. 置信度评估 - 动态评估每步推理的可信度
 * 4. Pre-LayerNorm架构 - 在子层之前应用层归一化
 * 
 * 本实现完全基于TinyAI框架的V2 API，不依赖任何V1组件。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Config implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 基础模型配置 ====================
    
    /** 词汇表大小，默认50257 */
    private int vocabSize = 50257;
    
    /** 最大位置数（序列长度），默认2048 */
    private int nPositions = 2048;
    
    /** 嵌入维度，默认768 */
    private int nEmbd = 768;
    
    /** Transformer层数，默认12 */
    private int nLayer = 12;
    
    /** 注意力头数，默认12 */
    private int nHead = 12;
    
    /** 前馈网络中间层维度，默认4倍嵌入维度 */
    private int nInner = 3072;
    
    /** 激活函数类型，默认"gelu" */
    private String activationFunction = "gelu";
    
    // ==================== 推理配置 ====================
    
    /** 最大推理步骤数，默认7步 */
    private int maxReasoningSteps = 7;
    
    /** 推理隐藏层维度，默认为嵌入维度的2倍 */
    private int reasoningHiddenDim = 1536;
    
    /** 推理置信度阈值，默认0.7 */
    private double confidenceThreshold = 0.7;
    
    // ==================== 反思配置 ====================
    
    /** 反思模块隐藏层维度，默认为嵌入维度的2倍 */
    private int reflectionHiddenDim = 1536;
    
    /** 反思质量分数维度，默认5个维度（逻辑性、完整性、正确性、清晰度、有用性） */
    private int qualityScoreDim = 5;
    
    /** 改进建议最大数量，默认3条 */
    private int maxSuggestions = 3;
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    private double residPdrop = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    private double embdPdrop = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    private double attnPdrop = 0.1;
    
    // ==================== 初始化配置 ====================
    
    /** 层归一化epsilon，默认1e-5 */
    private double layerNormEpsilon = 1e-5;
    
    /** 权重初始化范围，默认0.02 */
    private double initializerRange = 0.02;
    
    /**
     * 默认构造函数，创建标准DeepSeek-R1配置
     */
    public DeepSeekR1Config() {
        // 使用默认值
    }
    
    /**
     * 完整配置构造函数
     */
    public DeepSeekR1Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
                            int nHead, int nInner, String activationFunction,
                            int maxReasoningSteps, int reasoningHiddenDim, double confidenceThreshold,
                            int reflectionHiddenDim, int qualityScoreDim, int maxSuggestions,
                            double residPdrop, double embdPdrop, double attnPdrop,
                            double layerNormEpsilon, double initializerRange) {
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = nInner;
        this.activationFunction = activationFunction;
        this.maxReasoningSteps = maxReasoningSteps;
        this.reasoningHiddenDim = reasoningHiddenDim;
        this.confidenceThreshold = confidenceThreshold;
        this.reflectionHiddenDim = reflectionHiddenDim;
        this.qualityScoreDim = qualityScoreDim;
        this.maxSuggestions = maxSuggestions;
        this.residPdrop = residPdrop;
        this.embdPdrop = embdPdrop;
        this.attnPdrop = attnPdrop;
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-R1配置
     * 配置：768维, 12层, 12头, 序列长度2048
     */
    public static DeepSeekR1Config createStandardConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setVocabSize(50257);
        config.setNEmbd(768);
        config.setNLayer(12);
        config.setNHead(12);
        config.setNInner(3072);
        config.setNPositions(2048);
        config.setMaxReasoningSteps(7);
        config.setReasoningHiddenDim(1536);
        config.setReflectionHiddenDim(1536);
        return config;
    }
    
    /**
     * 创建微型DeepSeek-R1配置（用于快速测试）
     * 配置：256维, 6层, 8头, 序列长度512
     */
    public static DeepSeekR1Config createTinyConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setVocabSize(1000);
        config.setNEmbd(64);
        config.setNLayer(6);
        config.setNHead(8);
        config.setNInner(128);
        config.setNPositions(32);
        config.setMaxReasoningSteps(5);
        config.setReasoningHiddenDim(512);
        config.setReflectionHiddenDim(512);
        return config;
    }
    
    /**
     * 创建小型DeepSeek-R1配置（用于学习和实验）
     * 配置：512维, 8层, 8头, 序列长度1024
     */
    public static DeepSeekR1Config createSmallConfig() {
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setVocabSize(30000);
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setNPositions(1024);
        config.setMaxReasoningSteps(6);
        config.setReasoningHiddenDim(1024);
        config.setReflectionHiddenDim(1024);
        return config;
    }
    
    // ==================== 验证方法 ====================
    
    /**
     * 验证配置的有效性
     * 
     * @throws IllegalArgumentException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0，实际: " + vocabSize);
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0，实际: " + nPositions);
        }
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0，实际: " + nEmbd);
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("层数必须大于0，实际: " + nLayer);
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0，实际: " + nHead);
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException(
                String.format("嵌入维度(%d)必须能被注意力头数(%d)整除", nEmbd, nHead)
            );
        }
        if (nInner <= 0) {
            throw new IllegalArgumentException("前馈网络维度必须大于0，实际: " + nInner);
        }
        if (maxReasoningSteps <= 0) {
            throw new IllegalArgumentException("最大推理步骤数必须大于0，实际: " + maxReasoningSteps);
        }
        if (confidenceThreshold < 0 || confidenceThreshold > 1) {
            throw new IllegalArgumentException("置信度阈值必须在[0,1]范围内，实际: " + confidenceThreshold);
        }
        if (residPdrop < 0 || residPdrop >= 1) {
            throw new IllegalArgumentException("残差dropout概率必须在[0,1)范围内，实际: " + residPdrop);
        }
        if (embdPdrop < 0 || embdPdrop >= 1) {
            throw new IllegalArgumentException("嵌入dropout概率必须在[0,1)范围内，实际: " + embdPdrop);
        }
        if (attnPdrop < 0 || attnPdrop >= 1) {
            throw new IllegalArgumentException("注意力dropout概率必须在[0,1)范围内，实际: " + attnPdrop);
        }
    }
    
    /**
     * 估算模型参数数量
     * 
     * @return 估算的参数数量
     */
    public long estimateParameterCount() {
        // Token嵌入: vocabSize * nEmbd
        long tokenEmbed = (long) vocabSize * nEmbd;
        
        // 位置嵌入: nPositions * nEmbd
        long posEmbed = (long) nPositions * nEmbd;
        
        // 每个Transformer块的参数
        // 注意力: QKV投影(3 * nEmbd * nEmbd) + 输出投影(nEmbd * nEmbd) + 偏置(4 * nEmbd)
        long attnParams = 4L * nEmbd * nEmbd + 4L * nEmbd;
        
        // LayerNorm1: gamma(nEmbd) + beta(nEmbd)
        long ln1Params = 2L * nEmbd;
        
        // 前馈网络: fc1(nEmbd * nInner + nInner) + fc2(nInner * nEmbd + nEmbd)
        long ffnParams = (long) nEmbd * nInner + nInner + (long) nInner * nEmbd + nEmbd;
        
        // LayerNorm2: gamma(nEmbd) + beta(nEmbd)
        long ln2Params = 2L * nEmbd;
        
        // 每层总参数
        long paramsPerLayer = attnParams + ln1Params + ffnParams + ln2Params;
        
        // 所有Transformer层的参数
        long allLayersParams = paramsPerLayer * nLayer;
        
        // 推理模块参数
        // 推理投影: nEmbd * reasoningHiddenDim + reasoningHiddenDim
        // 推理输出: reasoningHiddenDim * nEmbd + nEmbd
        // 置信度评估: reasoningHiddenDim * 1 + 1
        long reasoningParams = (long) nEmbd * reasoningHiddenDim + reasoningHiddenDim +
                               (long) reasoningHiddenDim * nEmbd + nEmbd +
                               reasoningHiddenDim + 1;
        
        // 反思模块参数
        // 反思投影: nEmbd * reflectionHiddenDim + reflectionHiddenDim
        // 质量评分: reflectionHiddenDim * qualityScoreDim + qualityScoreDim
        // 改进建议: reflectionHiddenDim * nEmbd + nEmbd
        long reflectionParams = (long) nEmbd * reflectionHiddenDim + reflectionHiddenDim +
                                (long) reflectionHiddenDim * qualityScoreDim + qualityScoreDim +
                                (long) reflectionHiddenDim * nEmbd + nEmbd;
        
        // 最终LayerNorm: gamma(nEmbd) + beta(nEmbd)
        long finalLnParams = 2L * nEmbd;
        
        // 输出投影: nEmbd * vocabSize（通常与token嵌入权重共享）
        // 这里不重复计算
        
        // 总参数 = 嵌入 + 所有Transformer层 + 推理模块 + 反思模块 + 最终LN
        return tokenEmbed + posEmbed + allLayersParams + reasoningParams + reflectionParams + finalLnParams;
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getNPositions() {
        return nPositions;
    }
    
    public void setNPositions(int nPositions) {
        this.nPositions = nPositions;
    }
    
    public int getNEmbd() {
        return nEmbd;
    }
    
    public void setNEmbd(int nEmbd) {
        this.nEmbd = nEmbd;
    }
    
    public int getNLayer() {
        return nLayer;
    }
    
    public void setNLayer(int nLayer) {
        this.nLayer = nLayer;
    }
    
    public int getNHead() {
        return nHead;
    }
    
    public void setNHead(int nHead) {
        this.nHead = nHead;
    }
    
    public int getNInner() {
        return nInner;
    }
    
    public void setNInner(int nInner) {
        this.nInner = nInner;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getMaxReasoningSteps() {
        return maxReasoningSteps;
    }
    
    public void setMaxReasoningSteps(int maxReasoningSteps) {
        this.maxReasoningSteps = maxReasoningSteps;
    }
    
    public int getReasoningHiddenDim() {
        return reasoningHiddenDim;
    }
    
    public void setReasoningHiddenDim(int reasoningHiddenDim) {
        this.reasoningHiddenDim = reasoningHiddenDim;
    }
    
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    public void setConfidenceThreshold(double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
    
    public int getReflectionHiddenDim() {
        return reflectionHiddenDim;
    }
    
    public void setReflectionHiddenDim(int reflectionHiddenDim) {
        this.reflectionHiddenDim = reflectionHiddenDim;
    }
    
    public int getQualityScoreDim() {
        return qualityScoreDim;
    }
    
    public void setQualityScoreDim(int qualityScoreDim) {
        this.qualityScoreDim = qualityScoreDim;
    }
    
    public int getMaxSuggestions() {
        return maxSuggestions;
    }
    
    public void setMaxSuggestions(int maxSuggestions) {
        this.maxSuggestions = maxSuggestions;
    }
    
    public double getResidPdrop() {
        return residPdrop;
    }
    
    public void setResidPdrop(double residPdrop) {
        this.residPdrop = residPdrop;
    }
    
    public double getEmbdPdrop() {
        return embdPdrop;
    }
    
    public void setEmbdPdrop(double embdPdrop) {
        this.embdPdrop = embdPdrop;
    }
    
    public double getAttnPdrop() {
        return attnPdrop;
    }
    
    public void setAttnPdrop(double attnPdrop) {
        this.attnPdrop = attnPdrop;
    }
    
    public double getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(double layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public double getInitializerRange() {
        return initializerRange;
    }
    
    public void setInitializerRange(double initializerRange) {
        this.initializerRange = initializerRange;
    }
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekR1Config{\n" +
            "  vocabSize=%d,\n" +
            "  nPositions=%d,\n" +
            "  nEmbd=%d,\n" +
            "  nLayer=%d,\n" +
            "  nHead=%d,\n" +
            "  nInner=%d,\n" +
            "  activationFunction='%s',\n" +
            "  maxReasoningSteps=%d,\n" +
            "  reasoningHiddenDim=%d,\n" +
            "  confidenceThreshold=%.2f,\n" +
            "  reflectionHiddenDim=%d,\n" +
            "  qualityScoreDim=%d,\n" +
            "  maxSuggestions=%d,\n" +
            "  residPdrop=%.3f,\n" +
            "  embdPdrop=%.3f,\n" +
            "  attnPdrop=%.3f,\n" +
            "  layerNormEpsilon=%.1e,\n" +
            "  initializerRange=%.3f,\n" +
            "  estimatedParams=%s\n" +
            "}",
            vocabSize, nPositions, nEmbd, nLayer, nHead, nInner,
            activationFunction, maxReasoningSteps, reasoningHiddenDim, confidenceThreshold,
            reflectionHiddenDim, qualityScoreDim, maxSuggestions,
            residPdrop, embdPdrop, attnPdrop,
            layerNormEpsilon, initializerRange,
            formatParamCount(estimateParameterCount())
        );
    }
    
    /**
     * 格式化参数数量显示
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
