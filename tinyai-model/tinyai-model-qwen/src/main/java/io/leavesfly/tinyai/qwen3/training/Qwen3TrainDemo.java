package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.qwen3.Qwen3Config;
import io.leavesfly.tinyai.qwen3.Qwen3Model;

/**
 * Qwen3完整训练流程演示
 * 
 * 展示Qwen3模型的完整训练流程：
 * 1. 预训练 (Pretrain) - 因果语言建模
 * 2. 后训练 (Posttrain) - 指令微调
 * 3. 推理 (Inference) - 多种生成策略
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3TrainDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Qwen3 完整训练流程演示");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 示例1: 预训练
        example1_Pretrain();
        
        // 示例2: 后训练/微调
        example2_Posttrain();
        
        // 示例3: 推理
        example3_Inference();
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("演示完成！");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1: 预训练
     */
    private static void example1_Pretrain() {
        System.out.println("【示例1】预训练 (Pretrain)");
        System.out.println("-".repeat(80));
        
        try {
            // 创建小型模型用于演示
            Qwen3Config config = Qwen3Config.createSmallConfig();
            Qwen3Model model = new Qwen3Model("qwen3-pretrain", config);
            
            System.out.println("模型信息:");
            System.out.println("  - 参数量: " + formatParamCount(config.estimateParameterCount()));
            System.out.println("  - 隐藏维度: " + config.getHiddenSize());
            System.out.println("  - 层数: " + config.getNumHiddenLayers());
            
            // 创建演示数据集
            Qwen3Dataset trainDataset = Qwen3Dataset.createDemoDataset(
                config.getVocabSize(),
                100,  // 100个样本
                config.getMaxPositionEmbeddings(),
                2     // batch_size=2, 减小内存占用
            );
            
            // 创建预训练器
            Qwen3Pretrain pretrain = new Qwen3Pretrain(model, trainDataset);
            
            // 配置训练参数（演示用小epoch）
            pretrain.configure(
                2,           // maxEpochs
                2.5e-4f,     // learningRate
                50,          // warmupSteps
                1.0f         // maxGradNorm
            );
            
            pretrain.setCheckpoint("./checkpoints/qwen3_pretrain_demo", 500);
            pretrain.setLogInterval(5);  // 每5步输出一次日志，便于观察训练进度
            
            // 开始预训练
            System.out.println("\n开始预训练...");
            pretrain.train();
            
            // 打印统计
            Qwen3Pretrain.TrainingStats stats = pretrain.getStats();
            System.out.println("\n预训练统计:");
            System.out.println("  - 总步数: " + stats.totalSteps);
            System.out.println("  - 平均损失: " + String.format("%.4f", stats.avgLoss));
            
            System.out.println("\n✓ 预训练示例完成!");
            
        } catch (Exception e) {
            System.err.println("预训练失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 示例2: 后训练/微调
     */
    private static void example2_Posttrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("【示例2】后训练/微调 (Posttrain)");
        System.out.println("-".repeat(80));
        
        try {
            // 创建模型
            Qwen3Config config = Qwen3Config.createSmallConfig();
            Qwen3Model model = new Qwen3Model("qwen3-posttrain", config);
            
            // 创建训练和验证数据集
            Qwen3Dataset trainDataset = Qwen3Dataset.createDemoDataset(
                config.getVocabSize(), 80, config.getMaxPositionEmbeddings(), 2
            );
            Qwen3Dataset valDataset = Qwen3Dataset.createDemoDataset(
                config.getVocabSize(), 20, config.getMaxPositionEmbeddings(), 2
            );
            
            // 创建后训练器
            Qwen3Posttrain posttrain = new Qwen3Posttrain(model, trainDataset, valDataset);
            
            // 配置参数
            posttrain.configure(
                3,          // maxEpochs
                2.5e-5f,    // learningRate (预训练的1/10)
                2           // patience
            );
            
            // 开始微调
            System.out.println("\n开始微调...");
            posttrain.train();
            
            System.out.println("\n✓ 微调示例完成!");
            
        } catch (Exception e) {
            System.err.println("微调失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 示例3: 推理
     */
    private static void example3_Inference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("【示例3】推理 (Inference)");
        System.out.println("-".repeat(80));
        
        try {
            // 创建模型
            Qwen3Config config = Qwen3Config.createSmallConfig();
            Qwen3Model model = new Qwen3Model("qwen3-inference", config);
            
            // 创建推理器
            Qwen3Inference inference = new Qwen3Inference(model);
            
            // 准备输入
            int[] inputIds = {1, 2, 3, 4, 5};  // 示例输入
            int maxNewTokens = 10;
            
            System.out.println("输入token IDs: " + java.util.Arrays.toString(inputIds));
            System.out.println("最大生成token数: " + maxNewTokens);
            System.out.println();
            
            // 策略1: 贪婪解码
            System.out.println("1. 贪婪解码 (Greedy)");
            int[] greedyOutput = inference.generateGreedy(inputIds, maxNewTokens);
            System.out.println("   输出: " + java.util.Arrays.toString(greedyOutput));
            System.out.println();
            
            // 策略2: Top-K采样
            System.out.println("2. Top-K采样 (K=5)");
            int[] topKOutput = inference.generateTopK(inputIds, maxNewTokens, 5);
            System.out.println("   输出: " + java.util.Arrays.toString(topKOutput));
            System.out.println();
            
            // 策略3: Top-P采样
            System.out.println("3. Top-P采样 (P=0.9)");
            int[] topPOutput = inference.generateTopP(inputIds, maxNewTokens, 0.9f);
            System.out.println("   输出: " + java.util.Arrays.toString(topPOutput));
            System.out.println();
            
            // 策略4: 温度采样
            System.out.println("4. 温度采样 (T=0.8)");
            int[] tempOutput = inference.generateTemperature(inputIds, maxNewTokens, 0.8f);
            System.out.println("   输出: " + java.util.Arrays.toString(tempOutput));
            System.out.println();
            
            System.out.println("✓ 推理示例完成!");
            
        } catch (Exception e) {
            System.err.println("推理失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 格式化参数数量
     */
    private static String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
