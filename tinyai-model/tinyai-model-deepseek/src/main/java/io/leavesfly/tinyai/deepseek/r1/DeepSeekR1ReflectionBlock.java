package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.activation.Sigmoid;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * DeepSeek-R1反思模块（ReflectionBlock）
 * 
 * 实现自我评估和改进能力，包括：
 * 1. 质量评分 - 从多个维度评估推理质量
 * 2. 问题识别 - 发现推理过程中的潜在问题
 * 3. 改进建议 - 生成针对性的改进建议
 * 
 * 质量评分维度：
 * - 逻辑性：推理步骤的逻辑连贯性
 * - 完整性：是否考虑了所有相关因素
 * - 正确性：结论的准确性
 * - 清晰度：表达的清晰程度
 * - 有用性：对解决问题的帮助程度
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1ReflectionBlock extends Module {
    
    private final DeepSeekR1Config config;
    private final int qualityScoreDim;
    private final int maxSuggestions;
    
    // 反思投影层
    private final Linear reflectionProjection;
    private final LayerNorm reflectionLayerNorm;
    private final GELU reflectionActivation;
    
    // 质量评分层
    private final Linear qualityScoreProjection;
    private final Sigmoid qualityScoreSigmoid;
    
    // 改进建议生成层
    private final Linear suggestionProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1ReflectionBlock(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        this.qualityScoreDim = config.getQualityScoreDim();
        this.maxSuggestions = config.getMaxSuggestions();
        
        int dModel = config.getNEmbd();
        int hiddenDim = config.getReflectionHiddenDim();
        
        // 初始化反思投影层: [d_model] -> [hidden_dim]
        this.reflectionProjection = new Linear("reflection_proj", dModel, hiddenDim, true);
        this.reflectionLayerNorm = new LayerNorm("reflection_ln", hiddenDim, (float) config.getLayerNormEpsilon());
        this.reflectionActivation = new GELU("reflection_gelu");
        
        // 初始化质量评分层: [hidden_dim] -> [quality_score_dim]
        this.qualityScoreProjection = new Linear("quality_score_proj", hiddenDim, qualityScoreDim, true);
        this.qualityScoreSigmoid = new Sigmoid("quality_score_sigmoid");
        
        // 初始化改进建议生成层: [hidden_dim] -> [d_model]
        this.suggestionProjection = new Linear("suggestion_proj", hiddenDim, dModel, true);
        
        // 注册所有子模块
        registerModule("reflection_proj", reflectionProjection);
        registerModule("reflection_ln", reflectionLayerNorm);
        registerModule("reflection_gelu", reflectionActivation);
        registerModule("quality_score_proj", qualityScoreProjection);
        registerModule("quality_score_sigmoid", qualityScoreSigmoid);
        registerModule("suggestion_proj", suggestionProjection);
    }
    
    /**
     * 前向传播 - 执行反思评估
     * 
     * @param inputs 输入变量，inputs[0]为推理输出 [batch_size, seq_len, d_model]
     * @return 反思输出，包含质量评分和改进建议
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable reasoningOutput = inputs[0];
        
        // 执行反思评估
        ReflectionResult result = performReflection(reasoningOutput);
        
        // 返回改进建议（作为Variable）
        return result.suggestionOutput;
    }
    
    /**
     * 执行反思评估
     * 
     * @param reasoningOutput 推理输出 [batch_size, seq_len, d_model]
     * @return 反思结果对象
     */
    public ReflectionResult performReflection(Variable reasoningOutput) {
        // 1. 投影到反思隐藏空间
        Variable hidden = reflectionProjection.forward(reasoningOutput);
        hidden = reflectionLayerNorm.forward(hidden);
        hidden = reflectionActivation.forward(hidden);
        
        // 2. 计算质量评分
        QualityScore qualityScore = evaluateQuality(hidden);
        
        // 3. 生成改进建议
        Variable suggestionOutput = suggestionProjection.forward(hidden);
        
        return new ReflectionResult(qualityScore, suggestionOutput);
    }
    
    /**
     * 评估质量分数
     * 
     * @param hiddenState 反思隐藏状态 [batch_size, seq_len, hidden_dim]
     * @return 质量评分对象
     */
    private QualityScore evaluateQuality(Variable hiddenState) {
        // ✅ 使用Variable算子提取序列最后一个位置的隐藏状态
        NdArray hiddenData = hiddenState.getValue();
        int batchSize = hiddenData.getShape().getDimension(0);
        int seqLen = hiddenData.getShape().getDimension(1);
        int hiddenDim = hiddenData.getShape().getDimension(2);
        
        // 使用slice提取最后一个时间步: [batch_size, seq_len, hidden_dim] -> [batch_size, 1, hidden_dim]
        Variable lastHiddenVar = hiddenState.slice(
            new int[]{0, seqLen - 1, 0},
            new int[]{batchSize, seqLen, hiddenDim}
        );
        
        // 投影到质量分数维度
        Variable scoreVar = qualityScoreProjection.forward(lastHiddenVar);
        scoreVar = qualityScoreSigmoid.forward(scoreVar);
        
        // 提取各维度分数（对batch维度求平均）
        NdArray scoreData = scoreVar.getValue();
        double[] scores = new double[qualityScoreDim];
        for (int dim = 0; dim < qualityScoreDim; dim++) {
            double sum = 0.0;
            for (int b = 0; b < batchSize; b++) {
                sum += scoreData.get(b, 0, dim);
            }
            scores[dim] = sum / batchSize;
        }
        
        // 构建质量评分对象
        return new QualityScore(
            scores.length > 0 ? scores[0] : 0.0,  // 逻辑性
            scores.length > 1 ? scores[1] : 0.0,  // 完整性
            scores.length > 2 ? scores[2] : 0.0,  // 正确性
            scores.length > 3 ? scores[3] : 0.0,  // 清晰度
            scores.length > 4 ? scores[4] : 0.0   // 有用性
        );
    }
    
    /**
     * 反思结果类
     */
    public static class ReflectionResult {
        /** 质量评分 */
        public final QualityScore qualityScore;
        /** 改进建议输出 */
        public final Variable suggestionOutput;
        
        public ReflectionResult(QualityScore qualityScore, Variable suggestionOutput) {
            this.qualityScore = qualityScore;
            this.suggestionOutput = suggestionOutput;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReflectionResult{\n  %s\n}",
                qualityScore
            );
        }
    }
    
    /**
     * 质量评分类
     */
    public static class QualityScore {
        /** 逻辑性分数 [0, 1] */
        public final double logicScore;
        /** 完整性分数 [0, 1] */
        public final double completenessScore;
        /** 正确性分数 [0, 1] */
        public final double correctnessScore;
        /** 清晰度分数 [0, 1] */
        public final double clarityScore;
        /** 有用性分数 [0, 1] */
        public final double usefulnessScore;
        
        public QualityScore(double logicScore, double completenessScore, 
                           double correctnessScore, double clarityScore, 
                           double usefulnessScore) {
            this.logicScore = logicScore;
            this.completenessScore = completenessScore;
            this.correctnessScore = correctnessScore;
            this.clarityScore = clarityScore;
            this.usefulnessScore = usefulnessScore;
        }
        
        /**
         * 计算总体质量分数（各维度平均）
         */
        public double getOverallScore() {
            return (logicScore + completenessScore + correctnessScore + 
                    clarityScore + usefulnessScore) / 5.0;
        }
        
        @Override
        public String toString() {
            return String.format(
                "QualityScore{\n" +
                "    逻辑性: %.4f\n" +
                "    完整性: %.4f\n" +
                "    正确性: %.4f\n" +
                "    清晰度: %.4f\n" +
                "    有用性: %.4f\n" +
                "    总体评分: %.4f\n" +
                "  }",
                logicScore, completenessScore, correctnessScore,
                clarityScore, usefulnessScore, getOverallScore()
            );
        }
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
}
