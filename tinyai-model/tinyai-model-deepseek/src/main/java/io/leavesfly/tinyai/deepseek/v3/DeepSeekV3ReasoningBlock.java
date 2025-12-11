package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * DeepSeek-V3增强推理模块
 * 
 * V3的推理模块相比R1增加了任务感知能力和自我纠错机制。
 * 
 * 核心功能：
 * 1. 任务类型感知的推理策略
 * 2. 置信度评估
 * 3. 自我纠错机制（V3特有）
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3ReasoningBlock extends Module {
    
    private final DeepSeekV3Config config;
    
    // 推理投影层
    private Linear reasoningProjection;
    private GELU activation;
    private Linear reasoningOutput;
    
    // 置信度评估器
    private Linear confidenceEstimator;
    
    // 任务分类器
    private Linear taskClassifier;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3ReasoningBlock(String name, DeepSeekV3Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化组件
     */
    private void initializeComponents() {
        int nEmbd = config.getNEmbd();
        int reasoningHiddenDim = config.getReasoningHiddenDim();
        
        // 推理投影层: nEmbd -> reasoningHiddenDim -> nEmbd
        reasoningProjection = new Linear(
            name + "_reasoning_proj",
            nEmbd,
            reasoningHiddenDim,
            true
        );
        registerModule("reasoning_proj", reasoningProjection);
        
        activation = new GELU(name + "_gelu");
        registerModule("gelu", activation);
        
        reasoningOutput = new Linear(
            name + "_reasoning_out",
            reasoningHiddenDim,
            nEmbd,
            true
        );
        registerModule("reasoning_out", reasoningOutput);
        
        // 置信度评估器: reasoningHiddenDim -> 1
        confidenceEstimator = new Linear(
            name + "_confidence",
            reasoningHiddenDim,
            1,
            true
        );
        registerModule("confidence", confidenceEstimator);
        
        // 任务分类器: nEmbd -> numTaskTypes
        if (config.isEnableTaskAwareRouting()) {
            taskClassifier = new Linear(
                name + "_task_classifier",
                nEmbd,
                config.getNumTaskTypes(),
                true
            );
            registerModule("task_classifier", taskClassifier);
        }
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入张量 [batch_size, seq_len, nEmbd]
     * @return 推理输出 [batch_size, seq_len, nEmbd]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        
        // 推理计算
        Variable hidden = reasoningProjection.forward(input);
        hidden = activation.forward(hidden);
        Variable output = reasoningOutput.forward(hidden);
        
        return output;
    }
    
    /**
     * 带详细结果的推理
     * 
     * @param input 输入张量 [batch_size, seq_len, nEmbd]
     * @param taskType 任务类型（可选）
     * @return 推理结果
     */
    public ReasoningResult performReasoning(Variable input, TaskType taskType) {
        // 推理计算
        Variable hidden = reasoningProjection.forward(input);
        hidden = activation.forward(hidden);
        Variable output = reasoningOutput.forward(hidden);
        
        // 置信度评估
        Variable confidenceLogits = confidenceEstimator.forward(hidden);
        double confidence = computeAverageConfidence(confidenceLogits);
        
        // 任务类型识别（如果启用且未提供）
        TaskType detectedTaskType = taskType;
        if (detectedTaskType == null && config.isEnableTaskAwareRouting()) {
            detectedTaskType = detectTaskType(input);
        }
        
        // 自我纠错（如果启用）
        if (config.isEnableSelfCorrection() && confidence < config.getConfidenceThreshold()) {
            output = applySelfCorrection(output, input);
        }
        
        return new ReasoningResult(output, confidence, detectedTaskType);
    }
    
    /**
     * 检测任务类型
     */
    private TaskType detectTaskType(Variable input) {
        if (taskClassifier == null) {
            return TaskType.GENERAL;
        }
        
        // 对序列中的所有token进行分类，选择最频繁的类型
        Variable taskLogits = taskClassifier.forward(input);
        NdArray logitsArray = taskLogits.getValue();
        
        int batchSize = logitsArray.getShape().getDimension(0);
        int seqLen = logitsArray.getShape().getDimension(1);
        int numTaskTypes = logitsArray.getShape().getDimension(2);
        
        // 统计每种任务类型的得分
        float[] taskScores = new float[numTaskTypes];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int taskId = 0; taskId < numTaskTypes; taskId++) {
                    taskScores[taskId] += logitsArray.get(b, t, taskId);
                }
            }
        }
        
        // 选择得分最高的任务类型
        int maxTaskId = 0;
        float maxScore = taskScores[0];
        for (int i = 1; i < numTaskTypes; i++) {
            if (taskScores[i] > maxScore) {
                maxScore = taskScores[i];
                maxTaskId = i;
            }
        }
        
        return TaskType.fromId(maxTaskId);
    }
    
    /**
     * 计算平均置信度
     */
    private double computeAverageConfidence(Variable confidenceLogits) {
        NdArray logitsArray = confidenceLogits.getValue();
        int batchSize = logitsArray.getShape().getDimension(0);
        int seqLen = logitsArray.getShape().getDimension(1);
        
        double totalConfidence = 0.0;
        int count = 0;
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // Sigmoid激活
                double logit = logitsArray.get(b, t, 0);
                double conf = 1.0 / (1.0 + Math.exp(-logit));
                totalConfidence += conf;
                count++;
            }
        }
        
        return totalConfidence / count;
    }
    
    /**
     * 应用自我纠错机制
     */
    private Variable applySelfCorrection(Variable output, Variable originalInput) {
        // ✅ 使用Variable算术运算算子替代循环
        // 简化的自我纠错：将输出和原始输入进行加权组合
        // 置信度低时更多地保留原始输入信息
        float correctionWeight = 0.3f;
        
        // corrected = (1 - correctionWeight) * output + correctionWeight * originalInput
        // 使用Variable的算术运算算子
        Variable weight1 = new Variable(1.0f - correctionWeight);
        Variable weight2 = new Variable(correctionWeight);
        weight1.setRequireGrad(false);
        weight2.setRequireGrad(false);
        
        Variable scaledOutput = output.mul(weight1);
        Variable scaledInput = originalInput.mul(weight2);
        Variable corrected = scaledOutput.add(scaledInput);
        
        return corrected;
    }
    
    /**
     * 推理结果类
     */
    public static class ReasoningResult {
        /** 推理输出 */
        public final Variable reasoningOutput;
        /** 置信度 */
        public final double confidence;
        /** 检测到的任务类型 */
        public final TaskType taskType;
        
        public ReasoningResult(Variable reasoningOutput, double confidence, TaskType taskType) {
            this.reasoningOutput = reasoningOutput;
            this.confidence = confidence;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReasoningResult{confidence=%.4f, taskType=%s}",
                confidence,
                taskType != null ? taskType.getDescription() : "未知"
            );
        }
    }
}
