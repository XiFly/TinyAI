package io.leavesfly.tinyai.minimind.training.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.PretrainTrainer;
import io.leavesfly.tinyai.minimind.training.SFTTrainer;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import io.leavesfly.tinyai.minimind.training.dataset.PretrainDataset;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.minimind.training.dpo.DPOConfig;
import io.leavesfly.tinyai.minimind.training.dpo.DPOTrainer;
import io.leavesfly.tinyai.minimind.training.lora.LoRAConfig;
import io.leavesfly.tinyai.minimind.training.lora.LoRATrainer;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.util.*;

import static io.leavesfly.tinyai.minimind.training.demo.DemoConfig.*;

/**
 * MiniMind 训练演示 - 训练阶段执行器
 * 
 * 包含各训练阶段的执行逻辑：
 * - 步骤1: 无监督预训练
 * - 步骤2: 监督微调 (SFT)
 * - 步骤3: LoRA微调
 * - 步骤4: DPO训练
 * - 步骤5: 强化学习训练
 * - 步骤6: 推理测试
 * 
 * @author TinyAI Team
 */
public class DemoTrainingStages {

    // ========== 步骤1: 无监督预训练 ==========

    /**
     * 执行无监督预训练 - 使用标准 PretrainTrainer
     */
    public static MiniMindModel runUnsupervisedPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: MiniMind 无监督预训练 (Unsupervised Pretraining)");
        System.out.println("=".repeat(80));

        // 1. 创建分词器
        System.out.println("\n📝 创建分词器...");
        int maxSeqLen = 64;
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createSimpleTokenizer(maxSeqLen);
        setSharedTokenizer(tokenizer);
        System.out.println("  ✓ 分词器类型: 动态词汇表 (Simple-GPT1风格)");
        System.out.println("  ✓ 最大序列长度: " + maxSeqLen);

        // 2. 加载数据（动态构建词汇表）
        System.out.println("\n📝 准备预训练数据集...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        List<String> pretrainTexts = readFromFile(pretrainPath);
        
        int batchSize = 2;
        PretrainDataset dataset = new PretrainDataset(tokenizer, maxSeqLen, batchSize);
        dataset.loadFromTexts(pretrainTexts);
        dataset.prepare(true);
        
        // 冻结词汇表（类似GPT1 SimpleTokenizer）
        tokenizer.freeze();
        System.out.println("  ✓ 词汇表大小: " + tokenizer.getVocabSize() + " (已冻结)");
        System.out.println("  ✓ 预训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 创建MiniMind模型（使用实际词汇表大小）
        System.out.println("\n📝 创建MiniMind模型...");
        MiniMindConfig config = createMicroConfig(tokenizer.getVocabSize());
        MiniMindModel model = new MiniMindModel("minimind-pretrain", config);

        System.out.println("  ✓ 模型配置: Micro (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getHiddenSize());
        System.out.println("  ✓ 层数: " + config.getNumLayers());
        System.out.println("  ✓ 注意力头数: " + config.getNumHeads());

        // 4. 训练
        System.out.println("\n📝 开始无监督预训练...");
        System.out.println("  - 训练目标: 因果语言建模 (下一个词预测)");
        System.out.println("  - 学习率: 1e-2");
        System.out.println("  - 训练轮次: 3 epochs");
        System.out.println("-".repeat(80));

        PretrainTrainer trainer = new PretrainTrainer(model, dataset);
        trainer.configure(3, 1e-2f, 0, 1.0f);
        trainer.setLogInterval(10);
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 无监督预训练完成!");
        printPretrainSummary();

        return model;
    }

    // ========== 步骤2: 监督微调 ==========

    /**
     * 执行监督微调（SFT）
     */
    public static MiniMindModel runSupervisedFinetuning(MiniMindModel pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: MiniMind 监督微调 (Supervised Fine-tuning)");
        System.out.println("=".repeat(80));

        // 1. 加载数据
        System.out.println("\n📝 加载监督微调数据...");
        String trainPath = DATA_DIR + "/sft_train.txt";
        List<String> trainTexts = readFromFile(trainPath);
        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");

        // 2. 准备数据集
        System.out.println("\n📝 准备监督微调数据集...");
        MiniMindConfig config = pretrainedModel.getConfig();
        int batchSize = 2;
        
        SFTDataset dataset = new SFTDataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);
        for (String text : trainTexts) {
            dataset.addSample(text, "", text);
        }
        dataset.prepare(true);
        System.out.println("  ✓ 训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 训练
        System.out.println("\n📝 开始监督微调训练...");
        System.out.println("  - 训练目标: 指令跟随和对话生成");
        System.out.println("  - 学习率: 1e-3");
        System.out.println("  - 训练轮次: 3 epochs");
        System.out.println("-".repeat(80));

        SFTTrainer trainer = new SFTTrainer(pretrainedModel, dataset);
        trainer.configure(3, 1e-3f, 1.0f);
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 监督微调完成!");
        printSFTSummary();

        return pretrainedModel;
    }

    // ========== 步骤3: LoRA微调 ==========

    /**
     * 执行LoRA微调 - 参数高效微调
     */
    public static MiniMindModel runLoRAFinetuning(MiniMindModel sftModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🔧 步骤3: MiniMind LoRA微调 (Low-Rank Adaptation)");
        System.out.println("=".repeat(80));
        System.out.println("💡 LoRA核心思想: 冻结原始参数，只训练低秩分解矩阵");

        // 1. 配置LoRA
        System.out.println("\n📝 配置LoRA参数...");
        LoRAConfig loraConfig = new LoRAConfig();
        loraConfig.setRank(8);
        loraConfig.setAlpha(16.0f);
        loraConfig.setDropout(0.1f);
        loraConfig.setTargetModules(new String[]{"queryProj", "valueProj"});
        loraConfig.setFreezeOriginal(true);
        
        System.out.println("  ✓ LoRA秩 (r): " + loraConfig.getRank());
        System.out.println("  ✓ 缩放因子 (α): " + loraConfig.getAlpha());
        System.out.println("  ✓ 缩放系数 (α/r): " + loraConfig.getScaling());
        System.out.println("  ✓ 目标模块: " + String.join(", ", loraConfig.getTargetModules()));

        // 2. 注入 LoRA 层
        System.out.println("\n📝 注入 LoRA 层到模型...");
        int injectedCount = sftModel.applyLoRA(loraConfig);
        if (injectedCount > 0) {
            sftModel.printLoRAStats();
        }

        // 3. 准备数据
        System.out.println("\n📝 准备LoRA微调数据...");
        String trainPath = DATA_DIR + "/sft_train.txt";
        List<String> trainTexts = readFromFile(trainPath);
        
        MiniMindConfig config = sftModel.getConfig();
        int batchSize = 2;
        SFTDataset dataset = new SFTDataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);
        for (String text : trainTexts) {
            dataset.addSample(text, "", text);
        }
        dataset.prepare(true);
        System.out.println("  ✓ 训练样本数: " + dataset.getSampleCount());

        // 3. 训练
        System.out.println("\n📝 开始LoRA微调...");
        System.out.println("  - 学习率: 1e-4");
        System.out.println("  - 训练轮次: 2 epochs");
        System.out.println("-".repeat(80));

        LoRATrainer loraTrainer = new LoRATrainer(sftModel, dataset, loraConfig);
        loraTrainer.configure(2, 1e-4f, 1.0f);
        loraTrainer.printTrainableParams();
        loraTrainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ LoRA微调完成!");
        printLoRASummary();

        return sftModel;
    }

    // ========== 步骤4: DPO训练 ==========

    /**
     * 执行DPO训练 - 直接偏好优化
     */
    public static MiniMindModel runDPOTraining(MiniMindModel loraModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤4: MiniMind DPO训练 (Direct Preference Optimization)");
        System.out.println("=".repeat(80));
        System.out.println("💡 DPO核心思想: 无需奖励模型，直接从偏好对优化策略");

        // 1. 配置DPO
        System.out.println("\n📝 配置DPO参数...");
        DPOConfig dpoConfig = DPOConfig.createDefault();
        dpoConfig.setBeta(0.1f);
        dpoConfig.setLabelSmoothing(0.0f);
        dpoConfig.setUseLengthNormalization(false);
        dpoConfig.setResponseOnlyLoss(true);
        
        System.out.println("  ✓ Beta (β): " + dpoConfig.getBeta());
        System.out.println("  ✓ Response损失: " + dpoConfig.isResponseOnlyLoss());

        // 2. 加载数据
        System.out.println("\n📝 准备DPO偏好数据集...");
        String dpoPath = DATA_DIR + "/dpo_train.txt";
        List<String> dpoTexts = readFromFile(dpoPath);
        
        MiniMindConfig config = loraModel.getConfig();
        int batchSize = 1;
        DPODataset dpoDataset = new DPODataset(getSharedTokenizer(), config.getMaxSeqLen(), batchSize);
        
        for (String line : dpoTexts) {
            String[] parts = line.split("\\|\\|\\|");
            if (parts.length == 3) {
                dpoDataset.addSample(parts[0].trim(), parts[1].trim(), parts[2].trim());
            }
        }
        dpoDataset.prepare(true);
        System.out.println("  ✓ 偏好对数量: " + dpoDataset.getSampleCount());

        // 3. 训练
        System.out.println("\n📝 开始DPO训练...");
        System.out.println("  - 学习率: 1e-4");
        System.out.println("  - 训练轮次: 2 epochs");
        System.out.println("-".repeat(80));

        DPOTrainer dpoTrainer = new DPOTrainer(loraModel, dpoDataset, dpoConfig);
        dpoTrainer.configure(2, 1e-4f, 1.0f);  // 2 epochs, lr=1e-4 (教学小数据集需较高学习率)
        dpoTrainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ DPO训练完成!");
        printDPOSummary();

        return loraModel;
    }

    // ========== 步骤5: 强化学习训练 ==========

    /**
     * 执行强化学习训练（RLAIF）
     */
    public static MiniMindModel runReinforcementLearningTraining(MiniMindModel model) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤5: MiniMind 强化学习训练 (Reinforcement Learning)");
        System.out.println("=".repeat(80));
        System.out.println("💡 使用奖励加权的策略梯度方法优化模型");

        // 1. 加载数据
        System.out.println("\n📝 加载强化学习训练数据...");
        String rlPath = DATA_DIR + "/rl_train.txt";
        List<String> rlTexts = readFromFile(rlPath);
        System.out.println("  ✓ RL训练数据: " + rlTexts.size() + " 条");

        // 2. 解析数据
        System.out.println("\n📝 准备强化学习数据集...");
        List<String> texts = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        
        for (String line : rlTexts) {
            texts.add(removeRewardLabel(line));
            rewards.add(extractReward(line));
        }
        
        float avgReward = (float) rewards.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
        System.out.println("  ✓ RL样本数: " + texts.size());
        System.out.println("  ✓ 平均奖励: " + String.format("%.2f", avgReward));

        // 3. 训练配置
        MiniMindConfig config = model.getConfig();
        float learningRate = 5e-4f;
        int epochs = 2;
        int logInterval = 10;
        
        System.out.println("\n📝 开始强化学习训练...");
        System.out.println("  - 算法: 奖励加权策略梯度");
        System.out.println("  - 学习率: " + learningRate);
        System.out.println("  - 训练轮次: " + epochs);
        System.out.println("-".repeat(80));

        // 4. 训练循环
        Adam optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        SoftmaxCrossEntropy lossFunction = new SoftmaxCrossEntropy();
        model.setTraining(true);
        
        int step = 0;
        int maxSeqLen = config.getMaxSeqLen();
        MiniMindTokenizer tokenizer = getSharedTokenizer();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0.0f;
            int sampleCount = 0;
            
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                float reward = rewards.get(i);
                
                List<Integer> tokenIds = tokenizer.encode(text, true, true);
                if (tokenIds.size() < 2) continue;
                
                int seqLen = Math.min(tokenIds.size() - 1, maxSeqLen - 1);
                float[] inputData = new float[seqLen];
                float[] targetData = new float[seqLen];
                
                for (int j = 0; j < seqLen; j++) {
                    inputData[j] = tokenIds.get(j);
                    targetData[j] = tokenIds.get(j + 1);
                }
                
                Variable input = new Variable(NdArray.of(inputData, Shape.of(1, seqLen)));
                Variable target = new Variable(NdArray.of(targetData, Shape.of(1, seqLen)));
                
                Variable logits = model.predict(input);
                
                int[] logitsShape = logits.getValue().getShape().getShapeDims();
                int totalTokens = logitsShape[0] * logitsShape[1];
                int vocabSize = logitsShape[2];
                
                Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
                Variable targetReshaped = target.reshape(Shape.of(totalTokens, 1));
                
                Variable loss = lossFunction.loss(targetReshaped, logitsReshaped);
                Variable weightedLoss = loss.mul(new Variable(NdArray.of(reward)));
                
                model.clearGrads();
                weightedLoss.backward();
                optimizer.update();
                weightedLoss.unChainBackward();
                
                float lossValue = loss.getValue().getNumber().floatValue();
                epochLoss += lossValue * reward;
                sampleCount++;
                step++;
                
                if (step % logInterval == 0) {
                    System.out.printf("Epoch %d | Step %d | Loss: %.4f | Reward: %.2f%n",
                        epoch + 1, step, lossValue, reward);
                }
            }
            
            float avgLoss = sampleCount > 0 ? epochLoss / sampleCount : 0.0f;
            System.out.printf("Epoch %d 完成 | 平均加权损失: %.4f%n", epoch + 1, avgLoss);
        }

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 强化学习训练完成!");
        printRLSummary();

        return model;
    }

    // ========== 步骤6: 推理测试 ==========

    /**
     * 执行推理测试
     */
    public static void runInference(MiniMindModel model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤6: MiniMind 推理测试");
        System.out.println("=".repeat(80));

        model.setTraining(false);
        MiniMindTokenizer tokenizer = getSharedTokenizer();

        // 使用训练数据中出现过的短语作为 prompt，这样模型更容易续写
        List<String> testPrompts = Arrays.asList(
            "Machine learning is",
            "Neural networks are",
            "Deep learning",
            "AI technology"
        );

        System.out.println("\n📝 测试不同生成策略...");
        System.out.println("-".repeat(80));

        for (String prompt : testPrompts) {
            System.out.println("\n📌 Prompt: \"" + prompt + "\"");

            try {
                List<Integer> promptTokens = tokenizer.encode(prompt);
                int[] promptIds = promptTokens.stream().mapToInt(Integer::intValue).toArray();
                int promptLen = promptIds.length;

                // 1. 贪婪解码
                int[] greedyResult = model.generate(promptIds, 30, 0.0f, 0, 0.0f, 1.5f);
                String greedyGenerated = extractGenerated(tokenizer, greedyResult, promptLen);
                if (greedyGenerated.equals(" [无新生成]") || greedyGenerated.equals(" [空]")) {
                    // 贪婪解码生成EOS，尝试低温度采样
                    int[] fallbackResult = model.generate(promptIds, 30, 0.5f, 0, 0.0f, 1.5f);
                    greedyGenerated = extractGenerated(tokenizer, fallbackResult, promptLen);
                    System.out.println("  [Greedy→T=0.5] → " + prompt + greedyGenerated);
                } else {
                    System.out.println("  [Greedy]      → " + prompt + greedyGenerated);
                }

                // 2. Temperature 采样 (增加多样性)
                int[] tempResult = model.generate(promptIds, 30, 0.8f, 0, 0.0f);
                String tempGenerated = extractGenerated(tokenizer, tempResult, promptLen);
                System.out.println("  [Temp=0.8]    → " + prompt + tempGenerated);

                // 3. Top-K 采样
                int[] topkResult = model.generate(promptIds, 30, 1.0f, 10, 0.0f);
                String topkGenerated = extractGenerated(tokenizer, topkResult, promptLen);
                System.out.println("  [Top-K=10]    → " + prompt + topkGenerated);

                // 4. Top-P 采样 (Nucleus)
                int[] toppResult = model.generate(promptIds, 30, 1.0f, 0, 0.9f);
                String toppGenerated = extractGenerated(tokenizer, toppResult, promptLen);
                System.out.println("  [Top-P=0.9]   → " + prompt + toppGenerated);

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
        }

        System.out.println("\n" + "-".repeat(80));
        System.out.println("\n✅ 推理测试完成!");
        printInferenceSummary();
        printInferenceNotes();
    }

    /**
     * 从生成结果中提取新生成的部分
     */
    private static String extractGenerated(MiniMindTokenizer tokenizer, int[] result, int promptLen) {
        if (result.length <= promptLen) {
            return " [无新生成]";
        }
        // 提取 prompt 之后的 token
        int[] generatedIds = Arrays.copyOfRange(result, promptLen, result.length);
        List<Integer> generatedList = new ArrayList<>();
        for (int id : generatedIds) {
            generatedList.add(id);
        }
        String generated = tokenizer.decode(generatedList);
        if (generated.isEmpty()) {
            return " [空]";
        }
        // 确保生成内容与 prompt 之间有空格分隔
        if (!generated.startsWith(" ") && !generated.startsWith("\n")) {
            generated = " " + generated;
        }
        return generated;
    }

    /**
     * 打印推理注意事项
     */
    private static void printInferenceNotes() {
        System.out.println("\n💡 推理效果说明:");
        System.out.println("  ⚠ 当前为超小规模教学演示 (150条预训练+60条SFT数据)");
        System.out.println("  ⚠ 模型仅有0.4M参数，词汇表约600个token");
        System.out.println("  ⚠ 生成质量受限于数据量，主要用于理解训练流程");
        System.out.println("  ✓ 如需更好效果，请增加训练数据和训练轮次");
    }

    // ========== 阶段总结输出 ==========

    private static void printPretrainSummary() {
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和语法");
        System.out.println("  - 任务: 因果语言建模（预测下一个词）");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 技巧: 较高学习率 + 多轮训练");
    }

    private static void printSFTSummary() {
        System.out.println("\n💡 SFT阶段总结:");
        System.out.println("  - 目标: 学习遵循指令和生成高质量回答");
        System.out.println("  - 任务: 指令微调（问答对）");
        System.out.println("  - 数据: 带标签的指令-回答数据");
        System.out.println("  - 技巧: 小学习率 + 早停防止过拟合");
    }

    private static void printLoRASummary() {
        System.out.println("\n💡 LoRA阶段总结:");
        System.out.println("  - 目标: 在冻结原始参数的情况下进行低成本微调");
        System.out.println("  - 方法: 低秩分解 W = W0 + BA (只训练B和A矩阵)");
        System.out.println("  - 优势: 可训练参数量显著减少 (通常<1%)");
        System.out.println("  - 应用: 资源受限场景的模型定制");
    }

    private static void printDPOSummary() {
        System.out.println("\n💡 DPO阶段总结:");
        System.out.println("  - 目标: 使模型输出符合人类偏好");
        System.out.println("  - 方法: 直接从(prompt, chosen, rejected)三元组学习");
        System.out.println("  - 优势: 无需单独训练奖励模型，简化流程");
        System.out.println("  - 损失: L = -logσ(β(r_chosen - r_rejected))");
    }

    private static void printRLSummary() {
        System.out.println("\n💡 RL阶段总结:");
        System.out.println("  - 目标: 通过奖励信号对齐模型行为");
        System.out.println("  - 方法: 奖励加权的交叉熵损失");
        System.out.println("  - 效果: 高奖励样本获得更大梯度贡献");
        System.out.println("  - 技巧: 小学习率 + 奖励引导");
    }

    private static void printInferenceSummary() {
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词文本");
        System.out.println("  - 处理: 自回归生成");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/Top-K/Top-P");
    }
}
