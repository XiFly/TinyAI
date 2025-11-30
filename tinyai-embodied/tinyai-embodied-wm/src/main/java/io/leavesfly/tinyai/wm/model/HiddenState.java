package io.leavesfly.tinyai.wm.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 隐藏状态（Hidden State）
 * 表示RNN的隐藏状态，用于记忆和时序建模
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class HiddenState {
    
    /**
     * 隐藏状态向量 h
     * 形状: [hiddenSize]
     */
    private final NdArray h;
    
    /**
     * 细胞状态向量 c (LSTM专用，可选)
     * 形状: [hiddenSize]
     */
    private final NdArray c;
    
    /**
     * 时间戳
     */
    private final long timestamp;
    
    /**
     * 构造函数（LSTM，包含细胞状态）
     *
     * @param h 隐藏状态
     * @param c 细胞状态
     */
    public HiddenState(NdArray h, NdArray c) {
        this.h = h;
        this.c = c;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 构造函数（普通RNN/GRU，仅隐藏状态）
     *
     * @param h 隐藏状态
     */
    public HiddenState(NdArray h) {
        this(h, null);
    }
    
    /**
     * 创建零初始化的隐藏状态
     *
     * @param hiddenSize 隐藏状态维度
     * @param useLSTM 是否使用LSTM（需要细胞状态）
     */
    public static HiddenState zeros(int hiddenSize, boolean useLSTM) {
        NdArray h = NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(hiddenSize));
        if (useLSTM) {
            NdArray c = NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(hiddenSize));
            return new HiddenState(h, c);
        }
        return new HiddenState(h);
    }
    
    /**
     * 获取隐藏状态维度
     */
    public int getHiddenSize() {
        return h.getShape().getDimension(0);
    }
    
    /**
     * 是否使用LSTM（有细胞状态）
     */
    public boolean isLSTM() {
        return c != null;
    }
    
    // Getters
    public NdArray getH() {
        return h;
    }
    
    public NdArray getC() {
        return c;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 复制隐藏状态
     */
    public HiddenState copy() {
        NdArray hCopy = copyNdArray(h);
        NdArray cCopy = c != null ? copyNdArray(c) : null;
        return new HiddenState(hCopy, cCopy);
    }
    
    private NdArray copyNdArray(NdArray arr) {
        float[] data = arr.getArray();
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return NdArray.of(newData, arr.getShape());
    }
    
    @Override
    public String toString() {
        int[] hShape = h.getShape().getShapeDims();
        return String.format("HiddenState{h=[%d], c=%s}",
            hShape[0],
            c != null ? "[" + c.getShape().getDimension(0) + "]" : "null");
    }
}
