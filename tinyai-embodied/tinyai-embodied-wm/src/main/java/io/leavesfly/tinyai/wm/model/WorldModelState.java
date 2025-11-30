package io.leavesfly.tinyai.wm.model;

/**
 * 世界模型完整状态
 * 组合潜在状态和隐藏状态，表示世界模型的完整内部状态
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class WorldModelState {
    
    /**
     * 潜在状态（从观察编码而来）
     */
    private final LatentState latentState;
    
    /**
     * 隐藏状态（RNN记忆）
     */
    private final HiddenState hiddenState;
    
    /**
     * 时间戳
     */
    private final long timestamp;
    
    /**
     * 构造函数
     *
     * @param latentState 潜在状态
     * @param hiddenState 隐藏状态
     */
    public WorldModelState(LatentState latentState, HiddenState hiddenState) {
        this.latentState = latentState;
        this.hiddenState = hiddenState;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 创建初始状态
     *
     * @param latentSize 潜在状态维度
     * @param hiddenSize 隐藏状态维度
     * @param useLSTM 是否使用LSTM
     */
    public static WorldModelState createInitial(int latentSize, int hiddenSize, boolean useLSTM) {
        LatentState latentState = new LatentState(
            io.leavesfly.tinyai.ndarr.NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(latentSize))
        );
        HiddenState hiddenState = HiddenState.zeros(hiddenSize, useLSTM);
        return new WorldModelState(latentState, hiddenState);
    }
    
    /**
     * 获取状态总维度
     */
    public int getTotalDimension() {
        return latentState.getLatentSize() + hiddenState.getHiddenSize();
    }
    
    // Getters
    public LatentState getLatentState() {
        return latentState;
    }
    
    public HiddenState getHiddenState() {
        return hiddenState;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 复制世界模型状态
     */
    public WorldModelState copy() {
        return new WorldModelState(
            latentState.copy(),
            hiddenState.copy()
        );
    }
    
    @Override
    public String toString() {
        return String.format("WorldModelState{latent=%s, hidden=%s}",
            latentState, hiddenState);
    }
}
