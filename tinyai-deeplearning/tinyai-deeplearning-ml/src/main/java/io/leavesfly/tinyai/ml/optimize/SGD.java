package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.nnet.ParameterV1;

/**
 * 随机梯度下降优化器
 * 
 * 实现了经典的随机梯度下降算法，用于更新模型参数。
 * 更新公式：θ = θ - lr * ∇J(θ)
 * 
 * @author TinyDL
 * @version 1.0
 */
public class SGD extends Optimizer {

    private float lr;

    /**
     * 构造函数
     * @param target 目标模型
     * @param learnRate 学习率
     */
    public SGD(Model target, float learnRate) {
        super(target);
        lr = learnRate;
    }

    @Override
    public void updateOne(ParameterV1 parameterV1) {
        // 检查参数的梯度是否为null，如果为null则跳过更新
        if (parameterV1.getGrad() == null) {
            return;
        }
        parameterV1.setValue(parameterV1.getValue().sub(parameterV1.getGrad().mulNum(lr)));
    }
    
    /**
     * 设置学习率（用于学习率调度器）
     * 
     * @param learningRate 新的学习率
     */
    public void setLearningRate(float learningRate) {
        this.lr = learningRate;
    }
    
    /**
     * 获取当前学习率
     * 
     * @return 当前学习率
     */
    public float getLearningRate() {
        return lr;
    }
}