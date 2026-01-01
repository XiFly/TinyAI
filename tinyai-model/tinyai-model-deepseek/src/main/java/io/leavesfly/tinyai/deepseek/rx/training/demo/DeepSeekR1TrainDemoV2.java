package io.leavesfly.tinyai.deepseek.rx.training.demo;

import io.leavesfly.tinyai.deepseek.rx.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.rx.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.rx.training.*;
import io.leavesfly.tinyai.deepseek.rx.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.deepseek.rx.training.dataset.DeepSeekR1RLVRDataset;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-R1完整训练演示 V2版本
 * 
 * 提供完整的DeepSeek-R1训练流程编排：
 * 1. 数据准备阶段 - 生成训练数据集
 * 2. 预训练阶段 - 基础语言建模训练
 * 3. 后训练阶段 - 任务特定微调
 * 4. RLHF训练阶段 - 人类反馈强化学习
 * 5. RLVR训练阶段 - 可验证奖励强化学习
 * 6. 推理测试阶段 - 多种生成策略演示
 * 
 * 架构优化：
 * - 数据生成逻辑拆分到 {@link DeepSeekR1DatasetGenerator}
 * - 分词工具提取到 {@link DeepSeekR1TokenizerUtil}
 * - 主流程保持简洁，聚焦训练流程编排
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1TrainDemoV2 {
    
    private static DeepSeekR1TokenizerUtil sharedTokenizer = new DeepSeekR1TokenizerUtil();
    
    private static final String DATA_DIR = "./data/deepseek_r1_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/deepseek_r1_v2";
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("特色：推理增强 + 自我反思 + 强化学习对齐 (RLHF + RLVR)");
        System.out.println("=".repeat(80));
        
        try {
            // 步骤0: 准备数据集文件
            DeepSeekR1DatasetGenerator.prepareAllDatasets();
            
            // 步骤1: 预训练（无监督语言建模）
            DeepSeekR1Model pretrainedModel = runPretraining();
            
            // 步骤2: 后训练/微调（有监督学习）
            DeepSeekR1Model finetunedModel = runPosttraining(pretrainedModel);
            
            // 步骤3: 强化学习训练（RLHF - R1核心特色）
            DeepSeekR1Model rlhfModel = runRLHFTraining(finetunedModel);
            
            // 步骤4: 强化学习训练（RLVR - 可验证奖励训练）
//            DeepSeekR1Model alignedModel = runRLVRTraining(rlhfModel);
            
            // 步骤5: 推理测试
            runInference(rlhfModel);
            
            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ DeepSeek-R1完整训练流程演示成功!");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ========== 步骤1: 预训练 ==========
    
    /**
     * 执行预训练
     */
    private static DeepSeekR1Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: DeepSeek-R1 预训练 (Pretrain) - 无监督语言建模");
        System.out.println("=".repeat(80));
        
        // 1. 读取所有数据用于构建完整词汇表
        System.out.println("\n📝 加载所有数据以构建词汇表...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        String posttrainTrainPath = DATA_DIR + "/posttrain_train.txt";
        String posttrainValPath = DATA_DIR + "/posttrain_val.txt";
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        
        List<String> pretrainTexts = DeepSeekR1DatasetGenerator.readFromFile(pretrainPath);
        List<String> posttrainTrainTexts = DeepSeekR1DatasetGenerator.readFromFile(posttrainTrainPath);
        List<String> posttrainValTexts = DeepSeekR1DatasetGenerator.readFromFile(posttrainValPath);
        List<String> rlhfTexts = DeepSeekR1DatasetGenerator.readFromFile(rlhfPath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练训练数据: " + posttrainTrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练验证数据: " + posttrainValTexts.size() + " 条");
        System.out.println("  ✓ RLHF训练数据: " + rlhfTexts.size() + " 条");
        
        // 2. 基于所有数据构建完整词汇表
        System.out.println("\n📝 构建完整词汇表...");
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(posttrainTrainTexts);
        allTexts.addAll(posttrainValTexts);
        allTexts.addAll(rlhfTexts);
        
        // 遍历所有文本构建词汇表
        for (String text : allTexts) {
            String cleanText = DeepSeekR1TokenizerUtil.removeLabels(text);
            sharedTokenizer.encode(cleanText);
        }
        int vocabSize = sharedTokenizer.getVocabSize();
        
        // 冻结词汇表
        sharedTokenizer.freeze();
        
        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结,后续不再增加新词");
        
        // 3. 创建DeepSeek-R1模型
        System.out.println("\n📝 创建DeepSeek-R1模型...");
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        config.setVocabSize(vocabSize);
        config.setMaxReasoningSteps(2);  // 小规模演示使用较少推理步骤
        config.setNLayer(2);  // 减少层数加速训练
        
        DeepSeekR1Model model = new DeepSeekR1Model("deepseek-rx-pretrain-v2", config);
        
        System.out.println("  ✓ 模型配置: Tiny (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getNEmbd());
        System.out.println("  ✓ 层数: " + config.getNLayer());
        System.out.println("  ✓ 注意力头数: " + config.getNHead());
        System.out.println("  ✓ 最大推理步骤: " + config.getMaxReasoningSteps());
        System.out.println("  ✓ 质量评分维度: " + config.getQualityScoreDim());
        
        // 4. 准备数据集
        System.out.println("\n📝 准备训练数据集...");
        int seqLength = config.getNPositions();
        DeepSeekR1Dataset dataset = createDatasetFromTexts(
            pretrainTexts,
            seqLength,
            4,  // batch size
            config.getVocabSize()
        );
        
        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 4");
        System.out.println("  ✓ 序列长度: " + seqLength);
        
        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        DeepSeekR1Pretrain trainer = new DeepSeekR1Pretrain(model, dataset);
        trainer.configure(
            10,         // maxEpochs
            5e-2f,      // learningRate
            5,          // warmupSteps
            1.0f        // maxGradNorm
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 200);
        trainer.setLogInterval(50);
        trainer.configureParallel(true, 4);
        
        System.out.println("  ✓ 最大轮次: 10");
        System.out.println("  ✓ 学习率: 5e-2");
        System.out.println("  ✓ Warmup步数: 5");
        System.out.println("  ✓ 并行训练: 已启用 (4线程)");
        
        // 6. 开始训练
        System.out.println("\n📝 开始预训练...");
        System.out.println("-".repeat(80));
        trainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 预训练完成!");
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和推理基础");
        System.out.println("  - 任务: 因果语言建模（预测下一个词）");
        System.out.println("  - 数据: 大规模无标注文本（推理、数学、逻辑）");
        System.out.println("  - R1特色: 同时学习推理和反思能力");
        
        return model;
    }
    
    // ========== 步骤2: 后训练/微调 ==========
    
    /**
     * 执行后训练/微调
     */
    private static DeepSeekR1Model runPosttraining(DeepSeekR1Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: DeepSeek-R1 后训练/微调 (Posttrain) - 有监督学习");
        System.out.println("=".repeat(80));
        
        // 1. 加载后训练数据
        System.out.println("\n📝 加载后训练数据...");
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        String valPath = DATA_DIR + "/posttrain_val.txt";
        
        List<String> trainTexts = DeepSeekR1DatasetGenerator.readFromFile(trainPath);
        List<String> valTexts = DeepSeekR1DatasetGenerator.readFromFile(valPath);
        
        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 验证集: " + valTexts.size() + " 条");
        
        // 2. 准备数据集
        System.out.println("\n📝 准备后训练数据集...");
        DeepSeekR1Config config = pretrainedModel.getConfig();
        
        DeepSeekR1Dataset trainDataset = createDatasetFromTexts(
            trainTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize()
        );
        
        DeepSeekR1Dataset valDataset = createDatasetFromTexts(
            valTexts,
            config.getNPositions(),
            1,  // batch size
            config.getVocabSize()
        );
        
        System.out.println("  ✓ 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + valDataset.getSampleCount());
        
        // 3. 配置后训练器
        System.out.println("\n📝 配置后训练器...");
        DeepSeekR1Posttrain posttrain = new DeepSeekR1Posttrain(
            pretrainedModel,
            trainDataset,
            valDataset
        );
        
        posttrain.configure(
            3,          // maxEpochs
            1e-3f,      // learningRate
            2           // patience
        );
        
        System.out.println("  ✓ 最大轮次: 3");
        System.out.println("  ✓ 学习率: 1e-3");
        System.out.println("  ✓ 早停耐心值: 2");
        
        // 4. 开始后训练
        System.out.println("\n📝 开始后训练...");
        System.out.println("-".repeat(80));
        posttrain.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 后训练完成!");
        System.out.println("\n💡 后训练阶段总结:");
        System.out.println("  - 目标: 优化推理质量和反思能力");
        System.out.println("  - 任务: 任务特定的指令跟随");
        System.out.println("  - 数据: 带任务标签的推理问答对");
        System.out.println("  - 技巧: 小学习率 + 早停防止过拟合");
        System.out.println("  - R1特色: 增强链式推理和自我反思");
        
        return pretrainedModel;
    }
    
    // ========== 步骤3: 强化学习训练 ==========
    
    /**
     * 执行RLHF强化学习训练
     */
    private static DeepSeekR1Model runRLHFTraining(DeepSeekR1Model finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤3: DeepSeek-R1 RLHF训练 - 人类反馈强化学习");
        System.out.println("=".repeat(80));
        
        // 1. 加载RLHF数据
        System.out.println("\n📝 加载RLHF训练数据...");
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        List<String> rlhfTexts = DeepSeekR1DatasetGenerator.readFromFile(rlhfPath);
        
        System.out.println("  ✓ RLHF样本: " + rlhfTexts.size() + " 条");
        
        // 2. 准备RLHF数据集
        System.out.println("\n📝 准备RLHF数据集...");
        DeepSeekR1Config config = finetunedModel.getConfig();
        
        DeepSeekR1Dataset rlhfDataset = createRLHFDatasetFromTexts(
            rlhfTexts,
            config.getNPositions(),
            2,
            config.getVocabSize()
        );
        
        System.out.println("  ✓ RLHF训练样本: " + rlhfDataset.getSampleCount());
        
        // 3. 配置RLHF训练器
        System.out.println("\n📝 配置RLHF训练器...");
        DeepSeekR1RLHFTrainer rlhfTrainer = new DeepSeekR1RLHFTrainer(
            finetunedModel,
            rlhfDataset
        );
        
        rlhfTrainer.configure(
            2,          // maxEpochs
            5e-4f,      // learningRate
            1.0f,       // rewardWeight
            0.5f        // qualityWeight
        );
        
        System.out.println("  ✓ 最大轮次: 2");
        System.out.println("  ✓ 学习率: 5e-4");
        
        // 4. 开始RLHF训练
        System.out.println("\n📝 开始RLHF强化学习训练...");
        System.out.println("-".repeat(80));
        rlhfTrainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ RLHF训练完成!");
        System.out.println("\n💡 RLHF阶段总结:");
        System.out.println("  - 目标: 通过人类反馈对齐模型行为");
        System.out.println("  - 任务: 最大化人类偏好奖励");
        System.out.println("  - 数据: 带奖励标注的推理样本");
        System.out.println("  - R1特色: 平衡人类反馈与模型自评质量");
        
        return finetunedModel;
    }
    
    /**
     * 执行RLVR训练（可验证奖励强化学习）
     */
    private static DeepSeekR1Model runRLVRTraining(DeepSeekR1Model rlhfModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤4: DeepSeek-R1 RLVR训练 - 可验证奖励强化学习");
        System.out.println("=".repeat(80));
        
        // 1. 加载RLVR数据
        System.out.println("\n📝 加载RLVR训练数据...");
        String rlvrPath = DATA_DIR + "/rlvr_train.txt";
        List<String> rlvrTexts = DeepSeekR1DatasetGenerator.readFromFile(rlvrPath);
        
        System.out.println("  ✓ RLVR样本: " + rlvrTexts.size() + " 条");
        
        // 2. 准备RLVR数据集
        System.out.println("\n📝 准备RLVR数据集...");
        DeepSeekR1Config config = rlhfModel.getConfig();
        
        DeepSeekR1RLVRDataset rlvrDataset = createRLVRDatasetFromTexts(
            rlvrTexts,
            config.getNPositions(),
            2,
            config.getVocabSize()
        );
        
        System.out.println("  ✓ RLVR训练样本: " + rlvrDataset.getSampleCount());
        
        // 3. 配置RLVR训练器
        System.out.println("\n📝 配置RLVR训练器...");
        DeepSeekR1RLVRTrainer rlvrTrainer = new DeepSeekR1RLVRTrainer(
            rlhfModel,
            rlvrDataset
        );
        
        rlvrTrainer.configure(
            50,         // maxEpochs (增加训练轮次以充分学习)
            0.05f,      // learningRate (降低学习率提高稳定性)
            0.7f,       // correctnessWeight
            0.2f,       // reasoningQualityWeight
            0.1f        // verificationWeight
        );
        
        System.out.println("  ✓ 最大轮次: 50");
        System.out.println("  ✓ 学习率: 0.05");
        
        // 4. 开始RLVR训练
        System.out.println("\n📝 开始RLVR强化学习训练...");
        System.out.println("-".repeat(80));
        rlvrTrainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ RLVR训练完成!");
        System.out.println("\n💡 RLVR阶段总结:");
        System.out.println("  - 目标: 通过可验证标准优化正确性");
        System.out.println("  - 任务: 最大化二值验证奖励(0或1)");
        System.out.println("  - 数据: 带标准答案的可验证问题");
        System.out.println("  - 优势: RLHF + RLVR 结合提升模型能力");
        
        return rlhfModel;
    }
    
    // ========== 步骤5: 推理测试 ==========
    
    /**
     * 执行推理测试
     */
    private static void runInference(DeepSeekR1Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤5: DeepSeek-R1 推理与文本生成");
        System.out.println("=".repeat(80));
        
        // 1. 创建推理器
        System.out.println("\n📝 创建推理器...");
        DeepSeekR1Inference inference = new DeepSeekR1Inference(model);
        System.out.println("  ✓ 推理器准备完成");
        
        // 2. 测试用例
        String[] prompts = {
            "Reasoning requires",
            "Mathematics is",
            "Logic helps",
            "Self reflection"
        };
        
        System.out.println("\n📝 执行文本生成测试（带推理过程）...\n");
        
        for (int i = 0; i < prompts.length; i++) {
            String prompt = prompts[i];
            System.out.println("测试 " + (i + 1) + ": \"" + prompt + "\"");
            System.out.println("-".repeat(80));
            
            try {
                List<Integer> tokens = sharedTokenizer.encode(prompt);
                int[] promptIds = tokens.stream().mapToInt(Integer::intValue).toArray();
                
                // Greedy解码
                System.out.println("  策略1 [Greedy贪婪解码]: ");
                DeepSeekR1Inference.GenerationResult greedyResult = 
                    inference.generateGreedy(promptIds, 10);
                String greedyText = sharedTokenizer.decode(greedyResult.tokens);
                System.out.println("    → " + greedyText);
                // 调试：显示生成的token详情
                System.out.print("    Token IDs: ");
                for (int t : greedyResult.tokens) System.out.print(t + " ");
                System.out.println("(共" + greedyResult.tokens.length + "个)");
                
                // 打印推理统计
                if (!greedyResult.reasoningSteps.isEmpty()) {
                    DeepSeekR1Inference.ReasoningStep lastStep = 
                        greedyResult.reasoningSteps.get(greedyResult.reasoningSteps.size() - 1);
                    System.out.printf("    推理步骤: %d, 置信度: %.4f, 质量分: %.4f%n",
                        lastStep.reasoningSteps, lastStep.confidence, lastStep.qualityScore);
                }
                
                // Temperature采样
                System.out.println("  策略2 [Temperature=0.8]: ");
                DeepSeekR1Inference.GenerationResult tempResult = 
                    inference.generateWithTemperature(promptIds, 10, 0.8f);
                String tempText = sharedTokenizer.decode(tempResult.tokens);
                System.out.println("    → " + tempText);
                
            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
            
            System.out.println();
        }
        
        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词");
        System.out.println("  - 处理: 推理增强的自回归生成");
        System.out.println("  - 输出: 生成文本 + 推理过程");
        System.out.println("  - 策略: Greedy/Temperature采样");
        System.out.println("  - R1特色: 每个生成步骤都有推理置信度和质量评分");
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 从文本创建数据集
     */
    private static DeepSeekR1Dataset createDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize) {
        
        List<int[]> sequences = new ArrayList<>();
        
        for (String text : texts) {
            String cleanText = DeepSeekR1TokenizerUtil.removeLabels(text);
            
            // 编码文本
            List<Integer> tokens = sharedTokenizer.encode(cleanText);
            
            // 转换为数组
            int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
            
            // 截断或填充到maxSeqLength
            int[] paddedSeq = new int[maxSeqLength];
            Arrays.fill(paddedSeq, DeepSeekR1TokenizerUtil.PAD_TOKEN_ID);
            int copyLen = Math.min(sequence.length, maxSeqLength);
            System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
            
            sequences.add(paddedSeq);
        }
        
        return new DeepSeekR1Dataset(sequences, maxSeqLength, batchSize, true);
    }
    
    /**
     * 从RLHF文本创建数据集（包含奖励）
     */
    private static DeepSeekR1Dataset createRLHFDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize) {
        
        List<int[]> sequences = new ArrayList<>();
        List<String> reasoning = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        
        for (String text : texts) {
            // 提取奖励值
            float reward = DeepSeekR1TokenizerUtil.extractReward(text);
            String cleanText = DeepSeekR1TokenizerUtil.removeLabels(text);
            
            // 编码文本
            List<Integer> tokens = sharedTokenizer.encode(cleanText);
            
            // 转换为数组
            int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
            
            // 截断或填充
            int[] paddedSeq = new int[maxSeqLength];
            Arrays.fill(paddedSeq, DeepSeekR1TokenizerUtil.PAD_TOKEN_ID);
            int copyLen = Math.min(sequence.length, maxSeqLength);
            System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
            
            sequences.add(paddedSeq);
            reasoning.add(cleanText);
            rewards.add(reward);
        }
        
        return new DeepSeekR1Dataset(sequences, reasoning, rewards, 
                                     maxSeqLength, batchSize, true);
    }
    
    /**
     * 从RLVR文本创建数据集（包含验证类型和标准答案）
     */
    private static DeepSeekR1RLVRDataset createRLVRDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize) {
        
        DeepSeekR1RLVRDataset dataset = new DeepSeekR1RLVRDataset(
            batchSize, maxSeqLength, vocabSize
        );
        
        for (String text : texts) {
            // 解析格式: [TYPE:verifier_type] Question | GroundTruth
            String verifierType = DeepSeekR1TokenizerUtil.extractVerifierType(text);
            String cleanText = DeepSeekR1TokenizerUtil.removeLabels(text);
            
            // 分离问题和答案
            String[] parts = cleanText.split("\\|");
            if (parts.length >= 2) {
                String question = parts[0].trim();
                String groundTruth = parts[1].trim();
                
                // 编码问题
                List<Integer> tokens = sharedTokenizer.encode(question);
                
                // 转换为数组
                float[] tokenIds = new float[maxSeqLength];
                Arrays.fill(tokenIds, (float) DeepSeekR1TokenizerUtil.PAD_TOKEN_ID);
                int copyLen = Math.min(tokens.size(), maxSeqLength);
                for (int i = 0; i < copyLen; i++) {
                    tokenIds[i] = tokens.get(i).floatValue();
                }
                
                // 添加到数据集
                dataset.addSample(tokenIds, groundTruth, verifierType);
            }
        }
        
        return dataset;
    }
}
