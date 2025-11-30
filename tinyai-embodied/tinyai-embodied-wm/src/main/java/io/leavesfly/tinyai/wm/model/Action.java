package io.leavesfly.tinyai.wm.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 动作定义
 * 表示智能体在环境中可执行的动作
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class Action {
    
    /**
     * 动作向量
     * 形状: [actionSize]
     */
    private final NdArray actionVector;
    
    /**
     * 动作类型（离散或连续）
     */
    private final ActionType actionType;
    
    /**
     * 时间戳
     */
    private final long timestamp;
    
    /**
     * 构造函数
     *
     * @param actionVector 动作向量
     * @param actionType 动作类型
     */
    public Action(NdArray actionVector, ActionType actionType) {
        this.actionVector = actionVector;
        this.actionType = actionType;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 创建连续动作
     *
     * @param values 动作值数组
     */
    public static Action createContinuous(double... values) {
        float[] floatValues = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            floatValues[i] = (float) values[i];
        }
        NdArray vector = NdArray.of(floatValues, Shape.of(values.length));
        return new Action(vector, ActionType.CONTINUOUS);
    }
    
    /**
     * 创建离散动作
     *
     * @param actionIndex 动作索引
     * @param actionSize 动作空间大小
     */
    public static Action createDiscrete(int actionIndex, int actionSize) {
        float[] oneHot = new float[actionSize];
        oneHot[actionIndex] = 1.0f;
        NdArray vector = NdArray.of(oneHot, Shape.of(actionSize));
        return new Action(vector, ActionType.DISCRETE);
    }
    
    /**
     * 获取动作维度
     */
    public int getActionSize() {
        return actionVector.getShape().getDimension(0);
    }
    
    /**
     * 限制动作到有效范围 [-1, 1]
     */
    public Action clip() {
        if (actionType == ActionType.CONTINUOUS) {
            NdArray clipped = actionVector.clip(-1.0f, 1.0f);
            return new Action(clipped, actionType);
        }
        return this;
    }
    
    /**
     * 转换为数组
     */
    public double[] toArray() {
        float[] floatData = actionVector.getArray();
        double[] doubleData = new double[floatData.length];
        for (int i = 0; i < floatData.length; i++) {
            doubleData[i] = floatData[i];
        }
        return doubleData;
    }
    
    // Getters
    public NdArray getActionVector() {
        return actionVector;
    }
    
    public ActionType getActionType() {
        return actionType;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 复制动作
     */
    public Action copy() {
        // NdArray没有copy方法，手动复制
        float[] data = actionVector.getArray();
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        NdArray newVector = NdArray.of(newData, actionVector.getShape());
        return new Action(newVector, actionType);
    }
    
    @Override
    public String toString() {
        return String.format("Action{type=%s, vector=%s}",
            actionType,
            java.util.Arrays.toString(toArray()));
    }
}
