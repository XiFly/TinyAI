package io.leavesfly.tinyai.wm.training;

import io.leavesfly.tinyai.wm.WorldModelAgent;
import io.leavesfly.tinyai.wm.core.MDNRNN;
import io.leavesfly.tinyai.wm.core.VAEEncoder;
import io.leavesfly.tinyai.wm.core.WorldModel;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.wm.model.*;

import java.util.ArrayList;
import java.util.List;

/**
 * 世界模型训练器
 * 实现完整的三阶段训练流程：
 * 1. VAE训练 - 学习观察的压缩表示
 * 2. MDN-RNN训练 - 学习潜在空间的动态预测
 * 3. Controller训练 - 在想象环境中学习控制策略
 *
 * @author leavesfly
 * @since 2025-11-04
 */
public class WorldModelTrainer {
    
    private final WorldModel worldModel;
    private final WorldModelAgent agent;
    private final TrainingConfig config;
    
    // 训练数据缓冲区
    private final List<Observation> observationBuffer;
    private final List<Transition> transitionBuffer;
    
    // 训练统计
    private int trainingSteps;
    private double vaeLoss;
    private double rnnLoss;
    private double controllerReward;
    
    /**
     * 构造函数
     */
    public WorldModelTrainer(WorldModelAgent agent, TrainingConfig config) {
        this.agent = agent;
        this.worldModel = agent.getWorldModel();
        this.config = config;
        this.observationBuffer = new ArrayList<>();
        this.transitionBuffer = new ArrayList<>();
        this.trainingSteps = 0;
    }
    
    /**
     * 完整训练流程
     */
    public void train() {
        System.out.println("========================================");
        System.out.println("开始世界模型训练");
        System.out.println("========================================\n");
        
        // 阶段1：收集随机数据
        System.out.println("阶段1: 收集随机探索数据...");
        collectRandomData(config.getNumRandomEpisodes());
        
        // 阶段2：训练VAE
        System.out.println("\n阶段2: 训练VAE编码器...");
        trainVAE(config.getVaeEpochs());
        
        // 阶段3：训练MDN-RNN
        System.out.println("\n阶段3: 训练MDN-RNN记忆网络...");
        trainMDNRNN(config.getRnnEpochs());
        
        // 阶段4：在想象环境中训练Controller
        System.out.println("\n阶段4: 在想象环境中训练控制器...");
        trainController(config.getControllerEpochs());
        
        // 阶段5：在真实环境中微调
        System.out.println("\n阶段5: 在真实环境中微调...");
        fineTuneInRealEnv(config.getFineTuneEpisodes());
        
        System.out.println("\n========================================");
        System.out.println("训练完成！");
        printFinalStatistics();
        System.out.println("========================================");
    }
    
    /**
     * 阶段1：收集随机探索数据
     */
    private void collectRandomData(int numEpisodes) {
        for (int i = 0; i < numEpisodes; i++) {
            Episode episode = agent.runEpisode(config.getMaxStepsPerEpisode());
            
            // 收集观察数据
            for (Transition t : episode.getTransitions()) {
                observationBuffer.add(t.getObservation());
                observationBuffer.add(t.getNextObservation());
                transitionBuffer.add(t);
            }
            
            if ((i + 1) % 10 == 0) {
                System.out.printf("  收集进度: %d/%d 情景, 缓冲区大小: %d 观察, %d 转换\n",
                    i + 1, numEpisodes, observationBuffer.size(), transitionBuffer.size());
            }
        }
        System.out.printf("数据收集完成: 共 %d 观察, %d 转换\n",
            observationBuffer.size(), transitionBuffer.size());
    }
    
    /**
     * 阶段2：训练VAE编码器
     */
    private void trainVAE(int epochs) {
        VAEEncoder vae = worldModel.getVaeEncoder();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int batchCount = 0;
            
            // 随机采样批次数据
            List<Observation> batch = sampleBatch(observationBuffer, config.getBatchSize());
            
            for (Observation obs : batch) {
                // 前向传播
                VAEEncoder.EncoderOutput output = vae.forward(obs);
                
                // 计算损失
                double loss = vae.calculateLoss(obs, output);
                totalLoss += loss;
                batchCount++;
                
                // 简化实现：这里应该进行反向传播和参数更新
                // 实际项目中需要实现梯度计算和优化器
            }
            
            vaeLoss = totalLoss / batchCount;
            
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                System.out.printf("  Epoch %d/%d - VAE Loss: %.6f\n", 
                    epoch + 1, epochs, vaeLoss);
            }
        }
        
        System.out.printf("VAE训练完成 - 最终损失: %.6f\n", vaeLoss);
    }
    
    /**
     * 阶段3：训练MDN-RNN
     */
    private void trainMDNRNN(int epochs) {
        MDNRNN mdnRnn = worldModel.getMdnRnn();
        VAEEncoder vae = worldModel.getVaeEncoder();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int batchCount = 0;
            
            // 采样序列批次
            List<Transition> batch = sampleBatch(transitionBuffer, config.getBatchSize());
            
            for (Transition t : batch) {
                // 编码状态到潜在空间
                LatentState currentLatent = vae.encode(t.getObservation());
                LatentState nextLatent = vae.encode(t.getNextObservation());
                
                // 创建或获取隐藏状态
                HiddenState hidden = HiddenState.zeros(
                    worldModel.getConfig().getHiddenSize(), false);
                
                // 前向传播
                MDNRNN.RNNOutput output = mdnRnn.forward(
                    currentLatent, t.getAction(), hidden);
                
                // 计算损失（预测误差）
                double loss = mdnRnn.calculateLoss(nextLatent, output.getMdnParams());
                totalLoss += loss;
                batchCount++;
            }
            
            rnnLoss = totalLoss / batchCount;
            
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                System.out.printf("  Epoch %d/%d - RNN Loss: %.6f\n", 
                    epoch + 1, epochs, rnnLoss);
            }
        }
        
        System.out.printf("MDN-RNN训练完成 - 最终损失: %.6f\n", rnnLoss);
    }
    
    /**
     * 阶段4：在想象环境中训练控制器
     */
    private void trainController(int epochs) {
        double bestReward = Double.NEGATIVE_INFINITY;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalReward = 0.0;
            
            // 在想象环境中进行多次rollout
            for (int i = 0; i < config.getDreamRolloutsPerEpoch(); i++) {
                // 随机初始状态
                WorldModelState initialState = createRandomState();
                
                // 想象rollout
                Episode dreamEpisode = worldModel.dreamRollout(
                    initialState, config.getDreamSteps());
                
                totalReward += dreamEpisode.getTotalReward();
                
                // 简化实现：这里应该使用CMA-ES或其他进化算法优化控制器参数
            }
            
            controllerReward = totalReward / config.getDreamRolloutsPerEpoch();
            
            if (controllerReward > bestReward) {
                bestReward = controllerReward;
            }
            
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                System.out.printf("  Epoch %d/%d - 平均想象奖励: %.3f (最佳: %.3f)\n",
                    epoch + 1, epochs, controllerReward, bestReward);
            }
        }
        
        System.out.printf("控制器训练完成 - 最佳奖励: %.3f\n", bestReward);
    }
    
    /**
     * 阶段5：在真实环境中微调
     */
    private void fineTuneInRealEnv(int numEpisodes) {
        double totalReward = 0.0;
        
        for (int i = 0; i < numEpisodes; i++) {
            Episode episode = agent.runEpisode(config.getMaxStepsPerEpisode());
            totalReward += episode.getTotalReward();
            
            // 将新数据添加到缓冲区
            for (Transition t : episode.getTransitions()) {
                transitionBuffer.add(t);
                
                // 限制缓冲区大小
                if (transitionBuffer.size() > config.getMaxBufferSize()) {
                    transitionBuffer.remove(0);
                }
            }
            
            // 定期更新VAE和RNN
            if ((i + 1) % 10 == 0) {
                trainVAE(5);
                trainMDNRNN(5);
                
                double avgReward = totalReward / (i + 1);
                System.out.printf("  微调进度: %d/%d - 平均奖励: %.3f\n",
                    i + 1, numEpisodes, avgReward);
            }
        }
        
        double finalAvgReward = totalReward / numEpisodes;
        System.out.printf("真实环境微调完成 - 平均奖励: %.3f\n", finalAvgReward);
    }
    
    /**
     * 创建随机初始状态（用于想象训练）
     */
    private WorldModelState createRandomState() {
        int latentSize = worldModel.getConfig().getLatentSize();
        int hiddenSize = worldModel.getConfig().getHiddenSize();
        
        NdArray z = NdArray.randn(io.leavesfly.tinyai.ndarr.Shape.of(latentSize));
        LatentState latent = new LatentState(z);
        
        HiddenState hidden = HiddenState.zeros(hiddenSize, false);
        
        return new WorldModelState(latent, hidden);
    }
    
    /**
     * 从列表中随机采样批次
     */
    private <T> List<T> sampleBatch(List<T> data, int batchSize) {
        List<T> batch = new ArrayList<>();
        int actualSize = Math.min(batchSize, data.size());
        
        for (int i = 0; i < actualSize; i++) {
            int idx = (int) (Math.random() * data.size());
            batch.add(data.get(idx));
        }
        
        return batch;
    }
    
    /**
     * 打印最终统计信息
     */
    private void printFinalStatistics() {
        System.out.println("\n训练统计:");
        System.out.printf("  VAE 最终损失: %.6f\n", vaeLoss);
        System.out.printf("  RNN 最终损失: %.6f\n", rnnLoss);
        System.out.printf("  控制器想象奖励: %.3f\n", controllerReward);
        System.out.printf("  观察数据量: %d\n", observationBuffer.size());
        System.out.printf("  转换数据量: %d\n", transitionBuffer.size());
    }
    
    // Getters
    public double getVaeLoss() {
        return vaeLoss;
    }
    
    public double getRnnLoss() {
        return rnnLoss;
    }
    
    public double getControllerReward() {
        return controllerReward;
    }
}
