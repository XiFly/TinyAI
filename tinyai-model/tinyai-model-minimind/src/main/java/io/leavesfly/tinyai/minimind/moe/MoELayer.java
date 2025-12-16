package io.leavesfly.tinyai.minimind.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * MoE Layer - 混合专家层
 * 
 * 整合ExpertRouter和多个ExpertNetwork,实现完整的MoE机制
 * 
 * 工作流程:
 * 1. Router计算Top-K专家和权重
 * 2. 将输入路由到选中的专家
 * 3. 专家并行处理
 * 4. 按权重加权合并输出
 * 
 * 核心公式:
 * output = Σ(w_i · Expert_i(x)) for i in Top-K
 * 
 * @author leavesfly
 * @since 2024
 */
public class MoELayer extends Module {
    
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private final int numExperts;
    private final int topK;
    
    private final ExpertRouter router;
    private final List<ExpertNetwork> experts;
    
    // 统计信息
    private long[] expertUsageCount;  // 每个专家被使用次数
    private long totalCalls;          // 总调用次数
    
    /**
     * 构造函数
     * 
     * @param inputDim 输入维度
     * @param hiddenDim 专家隐藏层维度
     * @param outputDim 输出维度
     * @param numExperts 专家数量
     * @param topK Top-K选择数量
     * @param noiseFactor 路由噪声因子
     */
    public MoELayer(int inputDim, int hiddenDim, int outputDim, 
                    int numExperts, int topK, float noiseFactor) {
        super("moe_layer");
        
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.numExperts = numExperts;
        this.topK = topK;
        
        // 创建Router
        this.router = new ExpertRouter(inputDim, numExperts, topK, noiseFactor);
        registerModule("router", router);
        
        // 创建Experts
        this.experts = new ArrayList<>(numExperts);
        for (int i = 0; i < numExperts; i++) {
            ExpertNetwork expert = new ExpertNetwork(i, inputDim, hiddenDim, outputDim);
            experts.add(expert);
            registerModule("expert_" + i, expert);
        }
        
        // 初始化统计信息
        this.expertUsageCount = new long[numExperts];
        this.totalCalls = 0;
    }
    
    /**
     * 前向传播(Variable版本,Module接口要求)
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardVar(inputs[0]);
    }
    
    /**
     * 前向传播(Function接口)
     * 
     * @param inputs 输入数组
     * @return 输出NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        return forwardVar(input).getValue();
    }
    
    /**
     * 前向传播(Variable版本)
     * 
     * @param input 输入 [batch_size, input_dim]
     * @return 输出 [batch_size, output_dim]
     */
    public Variable forwardVar(Variable input) {
        int batchSize = input.getShape().getDimension(0);
        
        // 1. Router计算Top-K专家和权重
        ExpertRouter.RouterOutput routerOutput = router.forwardRouter(input);
        int[][] topKIndices = routerOutput.getTopKIndices();
        float[][] topKWeights = routerOutput.getTopKWeights();
        
        // 2. 准备输出变量 (初始化为零)
        Variable output = Variable.zeros(Shape.of(batchSize, outputDim));
        
        // 3. 对每个样本路由到对应专家
        for (int b = 0; b < batchSize; b++) {
            // 提取当前样本: 使用 indexSelect
            Variable batchIndexVar = new Variable(NdArray.of(new float[]{b}));
            batchIndexVar.setRequireGrad(false);
            Variable sampleInput = input.indexSelect(0, batchIndexVar);  // [1, input_dim]
            
            // 累加Top-K专家的输出
            Variable sampleOutput = Variable.zeros(Shape.of(1, outputDim));
            
            for (int k = 0; k < topK; k++) {
                int expertIdx = topKIndices[b][k];
                float weight = topKWeights[b][k];
                
                // 专家前向传播
                ExpertNetwork expert = experts.get(expertIdx);
                Variable expertOutput = expert.forwardVar(sampleInput);  // [1, output_dim]
                
                // 加权累加: 使用 Variable 算子
                Variable weightVar = new Variable(weight);
                weightVar.setRequireGrad(false);
                Variable weightedOutput = expertOutput.mul(weightVar);
                sampleOutput = sampleOutput.add(weightedOutput);
                
                // 更新统计
                expertUsageCount[expertIdx]++;
            }
            
            // 将样本输出写入总输出 (使用 scatterAdd 或直接赋值)
            // 简化实现: 直接操作 NdArray (因为这是写入操作，不影响梯度)
            NdArray outputData = output.getValue();
            NdArray sampleOutputData = sampleOutput.getValue();
            float[] outputBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) outputData).buffer;
            float[] sampleBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) sampleOutputData).buffer;
            System.arraycopy(sampleBuffer, 0, outputBuffer, b * outputDim, outputDim);
        }
        
        totalCalls += batchSize;
        
        // 4. 返回结果
        return output;
    }
    
    /**
     * 获取负载均衡损失(由LoadBalanceLoss调用)
     * 
     * @param routerOutput Router输出
     * @return 负载均衡相关统计
     */
    public LoadBalanceStats getLoadBalanceStats(ExpertRouter.RouterOutput routerOutput) {
        float[][] allWeights = routerOutput.getAllWeights();
        int batchSize = allWeights.length;
        
        // 计算每个专家的重要性(importance)
        float[] importance = new float[numExperts];
        for (int b = 0; b < batchSize; b++) {
            for (int e = 0; e < numExperts; e++) {
                importance[e] += allWeights[b][e];
            }
        }
        
        // 归一化
        float importanceSum = 0.0f;
        for (float imp : importance) {
            importanceSum += imp;
        }
        if (importanceSum > 0) {
            for (int e = 0; e < numExperts; e++) {
                importance[e] /= importanceSum;
            }
        }
        
        // 计算每个专家的负载(load)
        int[][] topKIndices = routerOutput.getTopKIndices();
        float[] load = new float[numExperts];
        
        for (int b = 0; b < batchSize; b++) {
            for (int k = 0; k < topK; k++) {
                int expertIdx = topKIndices[b][k];
                load[expertIdx] += 1.0f;
            }
        }
        
        // 归一化
        float loadSum = 0.0f;
        for (float ld : load) {
            loadSum += ld;
        }
        if (loadSum > 0) {
            for (int e = 0; e < numExperts; e++) {
                load[e] /= loadSum;
            }
        }
        
        return new LoadBalanceStats(importance, load);
    }
    
    /**
     * 获取专家使用统计
     */
    public ExpertUsageStats getUsageStats() {
        float[] usageRate = new float[numExperts];
        
        if (totalCalls > 0) {
            for (int e = 0; e < numExperts; e++) {
                usageRate[e] = (float) expertUsageCount[e] / totalCalls;
            }
        }
        
        return new ExpertUsageStats(expertUsageCount.clone(), usageRate, totalCalls);
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        expertUsageCount = new long[numExperts];
        totalCalls = 0;
    }
    
    /**
     * 获取Router
     */
    public ExpertRouter getRouter() {
        return router;
    }
    
    /**
     * 获取专家列表
     */
    public List<ExpertNetwork> getExperts() {
        return new ArrayList<>(experts);
    }
    
    /**
     * 获取专家数量
     */
    public int getNumExperts() {
        return numExperts;
    }
    
    /**
     * 获取Top-K数量
     */
    public int getTopK() {
        return topK;
    }
    
    @Override
    public String toString() {
        return String.format("MoELayer(in=%d, hidden=%d, out=%d, experts=%d, topK=%d)",
            inputDim, hiddenDim, outputDim, numExperts, topK);
    }
    
    /**
     * 负载均衡统计
     */
    public static class LoadBalanceStats {
        private final float[] importance;  // 专家重要性
        private final float[] load;        // 专家负载
        
        public LoadBalanceStats(float[] importance, float[] load) {
            this.importance = importance;
            this.load = load;
        }
        
        public float[] getImportance() {
            return importance;
        }
        
        public float[] getLoad() {
            return load;
        }
    }
    
    /**
     * 专家使用统计
     */
    public static class ExpertUsageStats {
        private final long[] usageCount;   // 使用次数
        private final float[] usageRate;   // 使用率
        private final long totalCalls;     // 总调用次数
        
        public ExpertUsageStats(long[] usageCount, float[] usageRate, long totalCalls) {
            this.usageCount = usageCount;
            this.usageRate = usageRate;
            this.totalCalls = totalCalls;
        }
        
        public long[] getUsageCount() {
            return usageCount;
        }
        
        public float[] getUsageRate() {
            return usageRate;
        }
        
        public long getTotalCalls() {
            return totalCalls;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("ExpertUsageStats{\n");
            for (int i = 0; i < usageCount.length; i++) {
                sb.append(String.format("  Expert%d: count=%d, rate=%.2f%%\n",
                    i, usageCount[i], usageRate[i] * 100));
            }
            sb.append(String.format("  Total calls: %d\n}", totalCalls));
            return sb.toString();
        }
    }
}
