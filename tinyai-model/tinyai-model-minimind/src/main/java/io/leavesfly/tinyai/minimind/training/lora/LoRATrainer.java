package io.leavesfly.tinyai.minimind.training.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.MaskedSoftmaxCELoss;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v1.ParameterV1;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * LoRA训练器
 * 
 * 使用LoRA进行参数高效微调
 * 只训练LoRA参数,冻结原始模型参数
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoRATrainer {
    
    private final MiniMindModel model;
    private final SFTDataset dataset;
    private final LoRAConfig loraConfig;
    private final MaskedSoftmaxCELoss lossFunction;
    private final Adam optimizer;
    
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;
    
    private int currentEpoch;
    private int currentStep;
    private List<Float> lossHistory;
    
    /**
     * 构造函数
     */
    public LoRATrainer(MiniMindModel model, SFTDataset dataset, LoRAConfig loraConfig) {
        this.model = model;
        this.dataset = dataset;
        this.loraConfig = loraConfig;
        this.lossFunction = new MaskedSoftmaxCELoss();
        
        // 默认配置(LoRA通常使用更高的学习率)
        this.maxEpochs = 3;
        this.learningRate = 1e-4f;
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.saveInterval = 500;
        this.checkpointDir = "./lora_checkpoints";
        
        // 创建优化器(仅优化LoRA参数)
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.lossHistory = new ArrayList<>();
        
        // 冻结非LoRA参数
        freezeNonLoRAParams();
    }
    
    /**
     * 冻结非LoRA参数
     * 注意: 如果模型没有LoRA参数,则不冻结任何参数(退化为全参数微调)
     */
    private void freezeNonLoRAParams() {
        int frozenCount = 0;
        int loraCount = 0;
        int totalParams = 0;
        
        // 先统计LoRA参数数量
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            totalParams++;
            if (paramName.toLowerCase().contains("lora")) {
                loraCount++;
            }
        }
        
        // 如果没有LoRA参数,退化为全参数微调
        if (loraCount == 0) {
            System.out.println("⚠️ 未检测到LoRA参数,退化为全参数微调模式");
            System.out.println("可训练参数: " + totalParams);
            return;
        }
        
        // 存在LoRA参数时,冻结非LoRA参数
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            ParameterV1 param = entry.getValue();
            
            if (!paramName.toLowerCase().contains("lora")) {
                param.clearGrad();
                frozenCount++;
            }
        }
        
        System.out.println("冻结参数: " + frozenCount);
        System.out.println("LoRA可训练参数: " + loraCount);
    }
    
    /**
     * 配置训练参数
     */
    public LoRATrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        
        optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("开始LoRA微调");
        System.out.println("=".repeat(60));
        System.out.println("LoRA配置: " + loraConfig);
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("=".repeat(60));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("LoRA微调完成!");
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        model.setTraining(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNextBatch()) {
            SFTDataset.Batch batch = dataset.getNextBatch();
            float stepLoss = trainStep(batch);
            
            epochLoss += stepLoss;
            batchCount++;
            currentStep++;
            
            lossHistory.add(stepLoss);
            
            if (currentStep % logInterval == 0) {
                double avgLoss = lossHistory.stream()
                    .skip(Math.max(0, lossHistory.size() - logInterval))
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
                
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f%n",
                    currentEpoch + 1, maxEpochs, currentStep, avgLoss);
            }
            
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练一步
     */
    private float trainStep(SFTDataset.Batch batch) {
        NdArray inputArray = batch.getInput();
        NdArray labelArray = batch.getLabels();
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        
        // 前向传播
        Variable logits = model.predict(input);
        
        // 使用 MaskedSoftmaxCELoss 计算损失（内置处理 3D logits 和 mask）
        // labels: [batch, seqLen], logits: [batch, seqLen, vocabSize]
        Variable loss = lossFunction.loss(labels, logits);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 清除梯度
        model.clearGrads();
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 更新参数
        optimizer.update();
        
        // 断开计算图
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        // 计算所有可训练参数的梯度范数
        for (var entry : model.getAllParams().entrySet()) {
            ParameterV1 param = entry.getValue();
            
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var entry : model.getAllParams().entrySet()) {
                ParameterV1 param = entry.getValue();
                
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint() {
        String filename = String.format("lora_checkpoint_epoch%d_step%d.model", 
            currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
            System.out.println("LoRA检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取可训练参数统计
     */
    public void printTrainableParams() {
        int totalParams = 0;
        int loraParams = 0;
        
        for (var entry : model.getAllParams().entrySet()) {
            String paramName = entry.getKey();
            ParameterV1 param = entry.getValue();
            int paramCount = param.getValue().getShape().size();
            
            totalParams += paramCount;
            if (paramName.toLowerCase().contains("lora")) {
                loraParams += paramCount;
            }
        }
        
        // 如果没有LoRA参数,所有参数都是可训练的(全参数微调模式)
        int trainableParams = loraParams > 0 ? loraParams : totalParams;
        float percentage = (float) trainableParams / totalParams * 100;
        
        System.out.println("=".repeat(60));
        System.out.println("参数统计:");
        System.out.println("  总参数: " + totalParams);
        System.out.println("  可训练参数: " + trainableParams + (loraParams == 0 ? " (全参数微调)" : " (LoRA)"));
        System.out.println("  训练参数占比: " + String.format("%.2f%%", percentage));
        System.out.println("=".repeat(60));
    }
    
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    public void setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
    }
}
