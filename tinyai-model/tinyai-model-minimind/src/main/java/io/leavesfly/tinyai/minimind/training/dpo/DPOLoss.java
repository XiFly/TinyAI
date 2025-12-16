package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * DPO (Direct Preference Optimization) 损失函数
 * 
 * DPO损失公式:
 * L_DPO = -log(σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x))))
 * 
 * 其中:
 * - π_θ: 策略模型(被训练的模型)
 * - π_ref: 参考模型(冻结的模型)
 * - y_w: 更好的响应(chosen/winner)
 * - y_l: 较差的响应(rejected/loser)
 * - β: KL散度惩罚系数
 * - σ: Sigmoid函数
 * 
 * DPO直接优化偏好,无需奖励模型
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOLoss {
    
    private final float beta;
    private final float labelSmoothing;
    
    /**
     * 构造函数
     * 
     * @param beta KL散度惩罚系数
     * @param labelSmoothing 标签平滑系数
     */
    public DPOLoss(float beta, float labelSmoothing) {
        this.beta = beta;
        this.labelSmoothing = labelSmoothing;
    }
    
    /**
     * 计算DPO损失
     * 
     * @param chosenLogProbs 策略模型在chosen响应上的对数概率
     * @param rejectedLogProbs 策略模型在rejected响应上的对数概率
     * @param refChosenLogProbs 参考模型在chosen响应上的对数概率
     * @param refRejectedLogProbs 参考模型在rejected响应上的对数概率
     * @return DPO损失
     */
    public Variable loss(Variable chosenLogProbs, Variable rejectedLogProbs,
                        Variable refChosenLogProbs, Variable refRejectedLogProbs) {
        
        // 计算策略模型的log ratio
        // log(π_θ(y_w|x)/π_θ(y_l|x)) = log π_θ(y_w|x) - log π_θ(y_l|x)
        Variable policyLogRatio = chosenLogProbs.sub(rejectedLogProbs);
        
        // 计算参考模型的log ratio
        // log(π_ref(y_w|x)/π_ref(y_l|x)) = log π_ref(y_w|x) - log π_ref(y_l|x)
        Variable refLogRatio = refChosenLogProbs.sub(refRejectedLogProbs);
        
        // 计算隐式奖励: β * [log(π_θ/π_ref)(y_w) - log(π_θ/π_ref)(y_l)]
        // = β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]
        Variable implicitReward = policyLogRatio.sub(refLogRatio);
        Variable scaledReward = implicitReward.mul(new Variable(NdArray.of(beta)));
        
        // 计算sigmoid损失: -log(σ(scaled_reward))
        // 等价于: log(1 + exp(-scaled_reward)) = softplus(-scaled_reward)
        Variable negScaledReward = scaledReward.mul(new Variable(NdArray.of(-1.0f)));
        Variable dpoLoss = softplus(negScaledReward);
        
        // 应用标签平滑
        if (labelSmoothing > 0) {
            // 添加正则化项鼓励chosen和rejected的logprobs都接近0
            Variable regularization = chosenLogProbs.add(rejectedLogProbs).mul(
                new Variable(NdArray.of(-labelSmoothing * 0.5f))
            );
            dpoLoss = dpoLoss.add(regularization);
        }
        
        // 返回平均损失
        return dpoLoss.mean(0, true);
    }
    
    /**
     * 计算log(sigmoid(x)) = -log(1 + exp(-x))
     * 
     * 使用数值稳定的实现:
     * log(sigmoid(x)) = -log(1 + exp(-x))
     *                 = -softplus(-x)
     *                 = x - softplus(x)  (当x > 0时)
     *                 = -softplus(-x)     (当x < 0时)
     * 
     * @param x 输入
     * @return log(sigmoid(x))
     */
    private Variable logSigmoid(Variable x) {
        // 使用log(sigmoid(x)) = -log(1 + exp(-x))的稳定实现
        // 等价于: x - softplus(x)
        return x.sub(softplus(x));
    }
    
    /**
     * Softplus函数: softplus(x) = log(1 + exp(x))
     * 简化实现,适用于TinyAI的Variable API
     * 
     * @param x 输入
     * @return softplus(x)
     */
    private Variable softplus(Variable x) {
        // softplus(x) = log(1 + exp(x))
        Variable expX = x.exp();
        Variable onePlusExp = expX.add(new Variable(NdArray.of(1.0f)));
        return onePlusExp.log();
    }
    
    /**
     * 计算序列的对数概率
     * 使用 mask 排除 prompt 部分，只计算 response 部分的对数概率
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]
     * @param labels 标签 [batch, seq_len]
     * @param mask 掩码 [batch, seq_len], 1表示计算(response)，0表示忽略(prompt)
     * @return 每个序列的平均对数概率
     */
    public Variable computeLogProbs(Variable logits, Variable labels, Variable mask) {
        // 获取维度信息
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        // 获取原始数据
        float[] logitsData = logits.getValue().getArray();
        float[] labelsData = labels.getValue().getArray();
        float[] maskData = mask.getValue().getArray();
        
        // 计算每个 token 的 log 概率
        float totalLogProb = 0.0f;
        int validTokens = 0;
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int flatIdx = b * seqLen + s;
                
                // 只计算 mask=1 的位置 (response 部分)
                if (maskData[flatIdx] > 0.5f) {
                    int labelIdx = (int) labelsData[flatIdx];
                    
                    // 提取当前位置的 logits
                    int logitsOffset = (b * seqLen + s) * vocabSize;
                    
                    // 计算 log softmax
                    float maxLogit = Float.NEGATIVE_INFINITY;
                    for (int v = 0; v < vocabSize; v++) {
                        maxLogit = Math.max(maxLogit, logitsData[logitsOffset + v]);
                    }
                    
                    float sumExp = 0.0f;
                    for (int v = 0; v < vocabSize; v++) {
                        sumExp += (float) Math.exp(logitsData[logitsOffset + v] - maxLogit);
                    }
                    float logSumExp = maxLogit + (float) Math.log(sumExp);
                    
                    // log 概率 = logit - log_sum_exp
                    float logProb = logitsData[logitsOffset + labelIdx] - logSumExp;
                    totalLogProb += logProb;
                    validTokens++;
                }
            }
        }
        
        // 返回平均 log 概率
        float avgLogProb = validTokens > 0 ? totalLogProb / validTokens : 0.0f;
        return new Variable(NdArray.of(avgLogProb));
    }
}
