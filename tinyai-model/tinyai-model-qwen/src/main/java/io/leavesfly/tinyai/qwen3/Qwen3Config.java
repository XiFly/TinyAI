package io.leavesfly.tinyai.qwen3;

/**
 * Qwen3模型配置类
 * 
 * Qwen3是基于现代Transformer架构的大语言模型，集成了以下先进技术：
 * 1. RMSNorm归一化 - 替代传统LayerNorm，提升计算效率
 * 2. 旋转位置编码(RoPE) - 支持长序列外推的相对位置编码
 * 3. 分组查询注意力(GQA) - 减少KV缓存内存占用
 * 4. SwiGLU激活函数 - 门控线性单元，增强非线性表达能力
 * 
 * 本实现完全基于TinyAI框架的V2 API，遵循Module-Variable设计模式。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Config {
    
    // ==================== 基础模型配置 ====================
    
    /** 词汇表大小，默认32000 */
    private int vocabSize = 32000;
    
    /** 隐藏层维度，默认2048 */
    private int hiddenSize = 2048;
    
    /** 中间层维度（FFN），默认为隐藏维度的2.75倍 */
    private int intermediateSize = 5632;
    
    /** Transformer解码器层数，默认24层 */
    private int numHiddenLayers = 24;
    
    /** 注意力头数，默认16 */
    private int numAttentionHeads = 16;
    
    /** 键值头数（用于分组查询注意力GQA），默认16 */
    private int numKeyValueHeads = 16;
    
    /** 最大位置编码数（序列长度），默认8192 */
    private int maxPositionEmbeddings = 8192;
    
    // ==================== 归一化配置 ====================
    
    /** RMSNorm的epsilon值，默认1e-6 */
    private double rmsNormEps = 1e-6;
    
    // ==================== 位置编码配置 ====================
    
    /** RoPE旋转位置编码的基础频率，默认10000.0 */
    private double ropeTheta = 10000.0;
    
    // ==================== 特殊标记配置 ====================
    
    /** 填充标记ID，默认0 */
    private int padTokenId = 0;
    
    /** 开始标记ID，默认1 */
    private int bosTokenId = 1;
    
    /** 结束标记ID，默认2 */
    private int eosTokenId = 2;
    
    /** 是否绑定词嵌入权重，默认false */
    private boolean tieWordEmbeddings = false;
    
    // ==================== 初始化配置 ====================
    
    /** 权重初始化标准差，默认0.02 */
    private double initializerRange = 0.02;
    
    /**
     * 默认构造函数
     */
    public Qwen3Config() {
        // 使用默认值
    }
    
    // ==================== 工厂方法 ====================
    
    /**
     * 创建小型Qwen3配置（用于快速测试）
     * 配置：256维, 2层, 4头, 序列长度128, 约2.6M参数
     * 设计用于轻量级测试，降低内存占用
     */
    public static Qwen3Config createSmallConfig() {
        Qwen3Config config = new Qwen3Config();
        config.setVocabSize(5000);     // 减小词表
        config.setHiddenSize(256);     // 减小隐藏维度
        config.setIntermediateSize(704);  // 256 * 2.75
        config.setNumHiddenLayers(2);  // 减少层数
        config.setNumAttentionHeads(4);
        config.setNumKeyValueHeads(4);
        config.setMaxPositionEmbeddings(128);  // 减小序列长度
        return config;
    }
    
    /**
     * 创建演示Qwen3配置
     * 配置：512维, 6层, 8头, 序列长度2048, 约62M参数
     */
    public static Qwen3Config createDemoConfig() {
        Qwen3Config config = new Qwen3Config();
        config.setVocabSize(32000);
        config.setHiddenSize(512);
        config.setIntermediateSize(1408);
        config.setNumHiddenLayers(6);
        config.setNumAttentionHeads(8);
        config.setNumKeyValueHeads(8);
        config.setMaxPositionEmbeddings(2048);
        return config;
    }
    
    /**
     * 创建标准Qwen3配置
     * 配置：2048维, 24层, 16头, 序列长度8192, 约1.8B参数
     */
    public static Qwen3Config createStandardConfig() {
        return new Qwen3Config(); // 使用默认值
    }
    
    // ==================== 计算属性 ====================
    
    /**
     * 获取每个注意力头的维度
     */
    public int getHeadDim() {
        return hiddenSize / numAttentionHeads;
    }
    
    /**
     * 获取键值组数（用于GQA）
     */
    public int getNumKeyValueGroups() {
        return numAttentionHeads / numKeyValueHeads;
    }
    
    // ==================== 验证方法 ====================
    
    /**
     * 验证配置有效性
     * 
     * @throws IllegalArgumentException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("隐藏层维度必须大于0");
        }
        if (numHiddenLayers <= 0) {
            throw new IllegalArgumentException("层数必须大于0");
        }
        if (numAttentionHeads <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0");
        }
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException(
                String.format("隐藏维度(%d)必须能被注意力头数(%d)整除", 
                    hiddenSize, numAttentionHeads)
            );
        }
        if (numKeyValueHeads <= 0) {
            throw new IllegalArgumentException("键值头数必须大于0");
        }
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                String.format("注意力头数(%d)必须能被键值头数(%d)整除", 
                    numAttentionHeads, numKeyValueHeads)
            );
        }
        if (intermediateSize <= 0) {
            throw new IllegalArgumentException("中间层维度必须大于0");
        }
        if (maxPositionEmbeddings <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0");
        }
    }
    
    /**
     * 估算模型参数数量
     */
    public long estimateParameterCount() {
        // Token嵌入: vocabSize * hiddenSize
        long tokenEmbed = (long) vocabSize * hiddenSize;
        
        // 每个Transformer块的参数
        // 注意力层: QKV投影 + O投影
        long qkvProj = (long) hiddenSize * (numAttentionHeads + 2 * numKeyValueHeads) * getHeadDim();
        long oProj = (long) hiddenSize * hiddenSize;
        long attnParams = qkvProj + oProj;
        
        // RMSNorm1: hiddenSize
        long rmsNorm1 = hiddenSize;
        
        // MLP: gate_proj + up_proj + down_proj
        long gateProj = (long) hiddenSize * intermediateSize;
        long upProj = (long) hiddenSize * intermediateSize;
        long downProj = (long) intermediateSize * hiddenSize;
        long mlpParams = gateProj + upProj + downProj;
        
        // RMSNorm2: hiddenSize
        long rmsNorm2 = hiddenSize;
        
        // 每层总参数
        long paramsPerLayer = attnParams + rmsNorm1 + mlpParams + rmsNorm2;
        
        // 所有层
        long allLayers = paramsPerLayer * numHiddenLayers;
        
        // 最终RMSNorm
        long finalNorm = hiddenSize;
        
        // 输出层（如果不绑定权重）
        long output = tieWordEmbeddings ? 0 : (long) hiddenSize * vocabSize;
        
        return tokenEmbed + allLayers + finalNorm + output;
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }
    
    public int getNumHiddenLayers() {
        return numHiddenLayers;
    }
    
    public void setNumHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
    }
    
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }
    
    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }
    
    public int getNumKeyValueHeads() {
        return numKeyValueHeads;
    }
    
    public void setNumKeyValueHeads(int numKeyValueHeads) {
        this.numKeyValueHeads = numKeyValueHeads;
    }
    
    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }
    
    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }
    
    public double getRmsNormEps() {
        return rmsNormEps;
    }
    
    public void setRmsNormEps(double rmsNormEps) {
        this.rmsNormEps = rmsNormEps;
    }
    
    public double getRopeTheta() {
        return ropeTheta;
    }
    
    public void setRopeTheta(double ropeTheta) {
        this.ropeTheta = ropeTheta;
    }
    
    public int getPadTokenId() {
        return padTokenId;
    }
    
    public void setPadTokenId(int padTokenId) {
        this.padTokenId = padTokenId;
    }
    
    public int getBosTokenId() {
        return bosTokenId;
    }
    
    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }
    
    public int getEosTokenId() {
        return eosTokenId;
    }
    
    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }
    
    public boolean isTieWordEmbeddings() {
        return tieWordEmbeddings;
    }
    
    public void setTieWordEmbeddings(boolean tieWordEmbeddings) {
        this.tieWordEmbeddings = tieWordEmbeddings;
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
            "Qwen3Config{\n" +
            "  vocabSize=%d,\n" +
            "  hiddenSize=%d,\n" +
            "  intermediateSize=%d,\n" +
            "  numHiddenLayers=%d,\n" +
            "  numAttentionHeads=%d,\n" +
            "  numKeyValueHeads=%d,\n" +
            "  maxPositionEmbeddings=%d,\n" +
            "  headDim=%d,\n" +
            "  numKeyValueGroups=%d,\n" +
            "  rmsNormEps=%.1e,\n" +
            "  ropeTheta=%.1f,\n" +
            "  estimatedParams=%s\n" +
            "}",
            vocabSize, hiddenSize, intermediateSize, numHiddenLayers,
            numAttentionHeads, numKeyValueHeads, maxPositionEmbeddings,
            getHeadDim(), getNumKeyValueGroups(), rmsNormEps, ropeTheta,
            formatParamCount(estimateParameterCount())
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
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
