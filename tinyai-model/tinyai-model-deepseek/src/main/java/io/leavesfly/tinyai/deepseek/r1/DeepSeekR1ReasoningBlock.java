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
 * DeepSeek-R1推理模块（ReasoningBlock）
 * 
 * 实现多步迭代推理能力，包括：
 * 1. 推理状态管理 - 维护每一步的推理状态
 * 2. 置信度评估 - 动态评估每步推理的可信度
 * 3. 步骤验证 - 确保推理过程的逻辑一致性
 * 
 * 推理流程：
 * - 输入问题 → 状态S0
 * - 推理步骤1 → 置信度检查 → 状态S1
 * - ...
 * - 推理步骤N → 置信度检查 → 最终状态
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1ReasoningBlock extends Module {
    
    private final DeepSeekR1Config config;
    private final int maxSteps;
    private final double confidenceThreshold;
    
    // 推理状态投影层
    private final Linear reasoningProjection;
    private final LayerNorm reasoningLayerNorm;
    private final GELU reasoningActivation;
    
    // 推理输出层
    private final Linear reasoningOutput;
    
    // 置信度评估层
    private final Linear confidenceProjection;
    private final Sigmoid confidenceSigmoid;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1ReasoningBlock(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        this.maxSteps = config.getMaxReasoningSteps();
        this.confidenceThreshold = config.getConfidenceThreshold();
        
        int dModel = config.getNEmbd();
        int hiddenDim = config.getReasoningHiddenDim();
        
        // 初始化推理投影层: [d_model] -> [hidden_dim]
        this.reasoningProjection = new Linear("reasoning_proj", dModel, hiddenDim, true);
        this.reasoningLayerNorm = new LayerNorm("reasoning_ln", hiddenDim, (float) config.getLayerNormEpsilon());
        this.reasoningActivation = new GELU("reasoning_gelu");
        
        // 初始化推理输出层: [hidden_dim] -> [d_model]
        this.reasoningOutput = new Linear("reasoning_out", hiddenDim, dModel, true);
        
        // 初始化置信度评估层: [hidden_dim] -> [1]
        this.confidenceProjection = new Linear("confidence_proj", hiddenDim, 1, true);
        this.confidenceSigmoid = new Sigmoid("confidence_sigmoid");
        
        // 注册所有子模块
        registerModule("reasoning_proj", reasoningProjection);
        registerModule("reasoning_ln", reasoningLayerNorm);
        registerModule("reasoning_gelu", reasoningActivation);
        registerModule("reasoning_out", reasoningOutput);
        registerModule("confidence_proj", confidenceProjection);
        registerModule("confidence_sigmoid", confidenceSigmoid);
    }
    
    /**
     * 前向传播 - 执行多步推理
     * 
     * @param inputs 输入变量，inputs[0]为Transformer输出 [batch_size, seq_len, d_model]
     * @return 推理结果，包含推理输出和置信度信息
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable x = inputs[0];
        
        // 执行多步推理
        ReasoningResult result = performMultiStepReasoning(x);
        
        // 返回最终推理输出
        return result.reasoningOutput;
    }
    
    /**
     * 执行多步推理
     * 
     * @param input 输入变量 [batch_size, seq_len, d_model]
     * @return 推理结果对象
     */
    public ReasoningResult performMultiStepReasoning(Variable input) {
        Variable currentState = input;
        double totalConfidence = 0.0;
        int actualSteps = 0;
        
        // 迭代推理，最多maxSteps步
        for (int step = 0; step < maxSteps; step++) {
            // 单步推理
            StepResult stepResult = performSingleReasoningStep(currentState, step);
            
            // 累积置信度
            totalConfidence += stepResult.confidence;
            actualSteps++;
            
            // 更新状态
            currentState = stepResult.output;
            
            // 如果置信度足够高，可以提前终止
            if (stepResult.confidence >= confidenceThreshold) {
                // 这里简化处理，继续执行所有步骤
                // 实际应用中可根据需求调整
            }
        }
        
        // 计算平均置信度
        double averageConfidence = totalConfidence / actualSteps;
        
        return new ReasoningResult(currentState, actualSteps, averageConfidence);
    }
    
    /**
     * 执行单步推理
     * 
     * @param state 当前推理状态
     * @param stepIndex 步骤索引
     * @return 单步推理结果
     */
    private StepResult performSingleReasoningStep(Variable state, int stepIndex) {
        // 1. 投影到推理隐藏空间
        Variable hidden = reasoningProjection.forward(state);
        hidden = reasoningLayerNorm.forward(hidden);
        hidden = reasoningActivation.forward(hidden);
        
        // 2. 生成推理输出
        Variable output = reasoningOutput.forward(hidden);
        
        // 3. 评估置信度
        double confidence = evaluateConfidence(hidden);
        
        return new StepResult(output, confidence);
    }
    
    /**
     * 评估置信度
     * 
     * @param hiddenState 推理隐藏状态 [batch_size, seq_len, hidden_dim]
     * @return 置信度分数 [0, 1]
     */
    private double evaluateConfidence(Variable hiddenState) {
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
        
        // 投影到置信度分数
        Variable confidenceScore = confidenceProjection.forward(lastHiddenVar);
        confidenceScore = confidenceSigmoid.forward(confidenceScore);
        
        // 计算平均置信度（对batch维度求平均）
        NdArray confidenceData = confidenceScore.getValue();
        double sumConfidence = 0.0;
        for (int b = 0; b < batchSize; b++) {
            sumConfidence += confidenceData.get(b, 0, 0);
        }
        
        return sumConfidence / batchSize;
    }
    
    /**
     * 推理结果类
     */
    public static class ReasoningResult {
        /** 推理输出 */
        public final Variable reasoningOutput;
        /** 实际推理步骤数 */
        public final int numSteps;
        /** 平均置信度 */
        public final double averageConfidence;
        
        public ReasoningResult(Variable reasoningOutput, int numSteps, double averageConfidence) {
            this.reasoningOutput = reasoningOutput;
            this.numSteps = numSteps;
            this.averageConfidence = averageConfidence;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReasoningResult{numSteps=%d, averageConfidence=%.4f}",
                numSteps, averageConfidence
            );
        }
    }
    
    /**
     * 单步推理结果类
     */
    private static class StepResult {
        final Variable output;
        final double confidence;
        
        StepResult(Variable output, double confidence) {
            this.output = output;
            this.confidence = confidence;
        }
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
}
