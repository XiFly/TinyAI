package io.leavesfly.tinyai.wm.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 观察状态
 * 表示智能体从环境获取的原始观察数据（如视觉图像）
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class Observation {
    
    /**
     * 视觉观察（图像）
     * 形状: [C, H, W] 例如 [3, 64, 64]
     */
    private final NdArray visualObservation;
    
    /**
     * 状态向量（车辆状态等）
     * 形状: [stateSize]
     */
    private final NdArray stateVector;
    
    /**
     * 时间戳
     */
    private final long timestamp;
    
    /**
     * 构造函数
     *
     * @param visualObservation 视觉观察
     * @param stateVector 状态向量
     */
    public Observation(NdArray visualObservation, NdArray stateVector) {
        this.visualObservation = visualObservation;
        this.stateVector = stateVector;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 构造函数（带时间戳）
     */
    public Observation(NdArray visualObservation, NdArray stateVector, long timestamp) {
        this.visualObservation = visualObservation;
        this.stateVector = stateVector;
        this.timestamp = timestamp;
    }
    
    /**
     * 获取视觉观察维度
     */
    public int[] getVisualShape() {
        return visualObservation.getShape().getShapeDims();
    }
    
    /**
     * 获取状态维度
     */
    public int getStateSize() {
        return stateVector.getShape().getDimension(0);
    }
    
    // Getters
    public NdArray getVisualObservation() {
        return visualObservation;
    }
    
    public NdArray getStateVector() {
        return stateVector;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 复制观察
     */
    public Observation copy() {
        return new Observation(
            NdArray.of(visualObservation.getArray(), visualObservation.getShape()),
            NdArray.of(stateVector.getArray(), stateVector.getShape()),
            timestamp
        );
    }
    
    @Override
    public String toString() {
        int[] vShape = visualObservation.getShape().getShapeDims();
        int[] sShape = stateVector.getShape().getShapeDims();
        return String.format("Observation{visual=[%d,%d,%d], state=[%d], timestamp=%d}",
            vShape[0], vShape[1], vShape[2],
            sShape[0],
            timestamp);
    }
}
