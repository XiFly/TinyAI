package io.leavesfly.tinyai.wm.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 潜在状态（Latent State）
 * 表示通过VAE编码后的压缩状态表示
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class LatentState {
    
    /**
     * 潜在向量 z
     * 形状: [latentSize]
     */
    private final NdArray z;
    
    /**
     * 均值向量 μ（用于VAE）
     */
    private final NdArray mu;
    
    /**
     * 对数方差 log(σ²)（用于VAE）
     */
    private final NdArray logVar;
    
    /**
     * 时间戳
     */
    private final long timestamp;
    
    /**
     * 构造函数（完整版）
     *
     * @param z 潜在向量
     * @param mu 均值向量
     * @param logVar 对数方差
     */
    public LatentState(NdArray z, NdArray mu, NdArray logVar) {
        this.z = z;
        this.mu = mu;
        this.logVar = logVar;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 构造函数（仅潜在向量）
     *
     * @param z 潜在向量
     */
    public LatentState(NdArray z) {
        this(z, null, null);
    }
    
    /**
     * 获取潜在向量维度
     */
    public int getLatentSize() {
        return z.getShape().getDimension(0);
    }
    
    /**
     * 计算KL散度（用于VAE训练）
     * KL(q||p) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
     */
    public double calculateKLDivergence() {
        if (mu == null || logVar == null) {
            return 0.0;
        }
        
        // 1 + log(σ²) - μ² - σ²
        NdArray muSquared = mu.mul(mu);
        NdArray expLogVar = logVar.exp();
        NdArray kl = muSquared.neg()
            .add(expLogVar.neg())
            .add(logVar)
            .add(NdArray.of(1.0f));
        
        // -0.5 * Σ(...)
        return -0.5 * kl.sum().getNumber().doubleValue();
    }
    
    // Getters
    public NdArray getZ() {
        return z;
    }
    
    public NdArray getMu() {
        return mu;
    }
    
    public NdArray getLogVar() {
        return logVar;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 复制潜在状态
     */
    public LatentState copy() {
        // 手动复制NdArray
        NdArray zCopy = copyNdArray(z);
        NdArray muCopy = mu != null ? copyNdArray(mu) : null;
        NdArray logVarCopy = logVar != null ? copyNdArray(logVar) : null;
        return new LatentState(zCopy, muCopy, logVarCopy);
    }
    
    private NdArray copyNdArray(NdArray arr) {
        float[] data = arr.getArray();
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return NdArray.of(newData, arr.getShape());
    }
    
    @Override
    public String toString() {
        return String.format("LatentState{z=%s, mu=%s, logVar=%s}",
            java.util.Arrays.toString(z.getShape().getShapeDims()),
            mu != null ? java.util.Arrays.toString(mu.getShape().getShapeDims()) : "null",
            logVar != null ? java.util.Arrays.toString(logVar.getShape().getShapeDims()) : "null");
    }
}
