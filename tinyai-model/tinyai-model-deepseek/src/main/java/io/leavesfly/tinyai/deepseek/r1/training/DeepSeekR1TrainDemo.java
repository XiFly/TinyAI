package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;

/**
 * DeepSeek-R1训练示例
 * 
 * 演示完整的训练流程：预训练 → 后训练 → 强化学习 → 推理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1TrainDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 完整训练流程演示");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 示例1: 预训练
        example1_Pretrain();
        
        // 示例2: 后训练/微调
        example2_Posttrain();
        
        // 示例3: 强化学习
        example3_RLHF();
        
        // 示例4: 推理
        example4_Inference();
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("所有训练示例完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1: 预训练
     */
    private static void example1_Pretrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例1: DeepSeek-R1预训练");
        System.out.println("=".repeat(80));
        
        // 创建微型模型用于演示，降低内存占用
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        // 减少推理步数以降低计算图深度
        config.setMaxReasoningSteps(2);
        config.setNLayer(2);  // 减少Transformer层数
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Tiny", config);
        
        // 创建示例数据集，减小序列长度
        int numSamples = 50;
        int seqLength = 16;
        int batchSize = 2;
        DeepSeekR1Dataset trainDataset = DeepSeekR1Dataset.createDummyDataset(
            numSamples, seqLength, config.getVocabSize(), batchSize
        );
        
        // 创建预训练器
        DeepSeekR1Pretrain pretrain = new DeepSeekR1Pretrain(model, trainDataset);
        
        // 配置训练参数
        pretrain.configure(
            2,           // maxEpochs (演示用小 epoch)
            1e-4f,       // learningRate
            50,          // warmupSteps
            1.0f         // maxGradNorm
        );
                
        // 设置较小的日志间隔以便观察训练进度
        pretrain.setLogInterval(5);
        
        pretrain.setCheckpoint("./checkpoints/r1_pretrain_demo", 500);
        
        // 开始预训练
        System.out.println("\n开始预训练...");
        pretrain.train();
        
        // 打印统计
        DeepSeekR1Pretrain.TrainingStats stats = pretrain.getStats();
        System.out.println("\n预训练统计: " + stats);
        
        System.out.println("\n✓ 预训练示例完成!");
    }
    
    /**
     * 示例2: 后训练/微调
     */
    private static void example2_Posttrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例2: DeepSeek-R1后训练/微调");
        System.out.println("=".repeat(80));
        
        // 创建微型模型用于演示，降低内存占用
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        // 减少推理步数以降低计算图深度
        config.setMaxReasoningSteps(2);
        config.setNLayer(2);  // 减少Transformer层数
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Tiny", config);
        
        // 创建训练和验证数据集，减小序列长度和批次大小以降低内存占用
        int numTrainSamples = 80;
        int numValSamples = 20;
        int seqLength = 16;
        int batchSize = 2;
        
        DeepSeekR1Dataset trainDataset = DeepSeekR1Dataset.createDummyDataset(
            numTrainSamples, seqLength, config.getVocabSize(), batchSize
        );
        DeepSeekR1Dataset valDataset = DeepSeekR1Dataset.createDummyDataset(
            numValSamples, seqLength, config.getVocabSize(), batchSize
        );
        
        // 创建后训练器
        DeepSeekR1Posttrain posttrain = new DeepSeekR1Posttrain(
            model, trainDataset, valDataset
        );
        
        // 配置
        posttrain.configure(
            3,           // maxEpochs
            1e-5f,       // learningRate (比预训练小)
            2            // patience
        );
        
        // 开始后训练
        System.out.println("\n开始后训练...");
        posttrain.train();
        
        System.out.println("\n✓ 后训练示例完成!");
    }
    
    /**
     * 示例3: 强化学习(RLHF)
     */
    private static void example3_RLHF() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例3: DeepSeek-R1强化学习训练(RLHF)");
        System.out.println("=".repeat(80));
        
        // 创建微型模型用于演示，降低内存占用
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        // 减少推理步数以降低计算图深度
        config.setMaxReasoningSteps(2);
        config.setNLayer(2);  // 减少Transformer层数
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Tiny", config);
        
        // 创建RLHF数据集(包含人类反馈)，减小序列长度和批次大小
        int numSamples = 50;
        int seqLength = 16;
        int batchSize = 2;
        
        DeepSeekR1Dataset rlhfDataset = DeepSeekR1Dataset.createDummyRLHFDataset(
            numSamples, seqLength, config.getVocabSize(), batchSize
        );
        
        // 创建RLHF训练器
        DeepSeekR1RLHFTrainer rlhfTrainer = new DeepSeekR1RLHFTrainer(model, rlhfDataset);
        
        // 配置
        rlhfTrainer.configure(
            2,           // maxEpochs
            5e-6f,       // learningRate (最小)
            1.0f,        // rewardWeight
            0.5f         // qualityWeight
        );
        
        // 开始RLHF训练
        System.out.println("\n开始RLHF训练...");
        rlhfTrainer.train();
        
        System.out.println("\n✓ RLHF示例完成!");
    }
    
    /**
     * 示例4: 推理
     */
    private static void example4_Inference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例4: DeepSeek-R1推理");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Tiny", config);
        
        // 创建推理引擎
        DeepSeekR1Inference inference = new DeepSeekR1Inference(model);
        
        // 准备提示词
        int[] promptIds = {1, 2, 3, 4, 5};
        int maxNewTokens = 10;
        
        System.out.println("\n提示词: " + java.util.Arrays.toString(promptIds));
        System.out.println("最大生成token数: " + maxNewTokens);
        
        // 贪婪解码
        System.out.println("\n--- 贪婪解码 ---");
        DeepSeekR1Inference.GenerationResult result1 = inference.generateGreedy(
            promptIds, maxNewTokens
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result1.tokens));
        result1.printReasoningTrace();
        
        // Temperature采样
        System.out.println("\n--- Temperature采样 (temperature=0.8) ---");
        DeepSeekR1Inference.GenerationResult result2 = inference.generateWithTemperature(
            promptIds, maxNewTokens, 0.8f
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result2.tokens));
        result2.printReasoningTrace();
        
        System.out.println("\n✓ 推理示例完成!");
    }
    
    /**
     * 示例5: 完整训练流水线
     */
    public static void example5_FullPipeline() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例5: 完整训练流水线");
        System.out.println("=".repeat(80));
        
        System.out.println("""
            完整训练流程:
            1. 预训练 (Pretrain)
               - 大规模无标注数据
               - 学习语言建模能力
               - 学习率: 1e-4
               
            2. 后训练 (Posttrain/Finetune)
               - 任务特定数据
               - 优化推理和反思质量
               - 学习率: 1e-5 (降低10倍)
               - 使用验证集和早停
               
            3. 强化学习 (RLHF)
               - 人类反馈数据
               - 最大化奖励函数
               - 学习率: 1e-6 (最小)
               - 平衡人类反馈和质量评分
               
            4. 推理部署
               - 多种生成策略
               - 推理过程可视化
               - 质量评分展示
            """);
        
        System.out.println("✓ 流水线说明完成!");
    }
}
