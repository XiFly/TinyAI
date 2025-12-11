package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * GPT-1训练和推理完整演示
 * <p>
 * 展示完整的训练流程:
 * 1. 预训练(Pretrain)
 * 2. 微调(Finetune/Posttrain)
 * 3. 推理(Inference)
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT1TrainDemo {

    private static List<String> preTrainTexts = Arrays.asList(
            // 深度学习基础
            "Deep learning is a subset of machine learning",
            "Deep learning uses neural networks with multiple layers",
            "Deep learning can learn complex patterns from data",
            "Deep learning is transforming artificial intelligence",
            "Deep learning models require large amounts of data",
            "Deep learning has achieved remarkable success in many fields",
            "Deep learning algorithms can automatically extract features",
            "Deep learning is the foundation of modern AI systems",
            "Deep learning enables end to end learning",
            "Deep learning models are trained on GPUs",
            // 机器学习
            "Machine learning algorithms improve with experience",
            "Machine learning is a branch of artificial intelligence",
            "Machine learning enables computers to learn from data",
            "Machine learning models can make predictions",
            "Machine learning requires feature engineering and data preprocessing",
            "Machine learning is used in recommendation systems",
            "Machine learning powers search engines and spam filters",
            "Supervised learning uses labeled training data",
            "Unsupervised learning finds patterns without labels",
            "Reinforcement learning learns through trial and error",
            // 神经网络
            "Neural networks learn patterns from data",
            "Neural networks have multiple layers of neurons",
            "Neural networks can approximate any function",
            "Neural networks are inspired by the human brain",
            "Neural networks consist of input output and hidden layers",
            "Neural networks use activation functions for nonlinearity",
            "Neural networks are trained using backpropagation",
            "Neural networks can process images text and speech",
            "Convolutional neural networks excel at image processing",
            "Recurrent neural networks handle sequential data",
            // 自然语言处理
            "Natural language processing enables computers to understand text",
            "Natural language processing is used in chatbots",
            "Natural language processing powers machine translation",
            "Natural language processing includes sentiment analysis",
            "Natural language processing helps computers read and write",
            "Language models can generate coherent text",
            "Language models learn from large text corpora",
            "Language models predict the next word in a sequence",
            "Language models are the core of modern NLP systems",
            "Word embeddings represent words as dense vectors",
            "Tokenization splits text into smaller units",
            "Text generation creates new content from prompts",
            // Transformer和GPT
            "Transformer architecture revolutionized NLP",
            "Transformer models use attention mechanisms",
            "Attention is all you need for sequence modeling",
            "GPT uses transformer decoder architecture",
            "GPT generates text in an autoregressive manner",
            "GPT learns from massive amounts of text data",
            "GPT can perform many NLP tasks without fine tuning",
            "The attention mechanism computes weighted relationships",
            "Self attention allows the model to focus on relevant parts",
            "Multi head attention captures different aspects",
            "Position embeddings encode sequence order",
            "Layer normalization stabilizes training",
            // 人工智能
            "Artificial intelligence is transforming the world",
            "Artificial intelligence can solve complex problems",
            "AI systems learn from experience and data",
            "AI is used in many applications today",
            "AI enables automation and intelligent decision making",
            "AI is reshaping industries and creating new opportunities",
            "AI research focuses on creating intelligent machines",
            "AI applications include robotics and autonomous vehicles",
            "General AI aims to match human intelligence",
            "Narrow AI excels at specific tasks",
            // 训练与优化
            "Training neural networks requires gradient descent",
            "Optimization algorithms minimize the loss function",
            "Batch size affects training speed and convergence",
            "Learning rate is a critical hyperparameter",
            "Regularization prevents overfitting",
            "Dropout is a common regularization technique",
            "Pretrained models can be fine tuned for specific tasks",
            "Transfer learning enables knowledge reuse",
            "Early stopping prevents overfitting during training",
            "Data augmentation increases training data variety",
            "Cross validation helps evaluate model performance",
            "Hyperparameter tuning optimizes model settings",
            // 应用场景
            "Image recognition uses convolutional neural networks",
            "Speech recognition converts audio to text",
            "Computer vision enables machines to see",
            "Text classification assigns labels to documents",
            "Named entity recognition extracts information from text",
            "Question answering systems provide accurate responses",
            "Sentiment analysis determines emotional tone",
            "Machine translation converts text between languages",
            "Object detection locates items in images",
            "Face recognition identifies people in photos",
            "Voice assistants use speech recognition and NLP",
            "Recommendation engines suggest relevant content",
            // 数据与特征
            "Data is the fuel for machine learning",
            "Feature extraction identifies important patterns",
            "Data cleaning removes noise and errors",
            "Feature scaling normalizes input values",
            "Dimensionality reduction simplifies complex data",
            "Data visualization helps understand patterns",
            "Labeled data is essential for supervised learning",
            "Big data enables training of large models",
            // 模型评估
            "Accuracy measures correct predictions",
            "Precision and recall evaluate classification",
            "Loss function quantifies prediction errors",
            "Validation data helps tune hyperparameters",
            "Test data evaluates final model performance",
            "Confusion matrix shows classification results",
            "ROC curve plots true and false positive rates",
            "F1 score balances precision and recall"
    );


    private static
    List<String> finetuneTexts = Arrays.asList(
            // 基础概念QA
            "Question: What is deep learning? Answer: Deep learning is a type of machine learning using neural networks",
            "Question: What is NLP? Answer: NLP stands for natural language processing",
            "Question: What is AI? Answer: AI is artificial intelligence",
            "Question: What are neural networks? Answer: Neural networks are computing systems inspired by the brain",
            "Question: What is machine learning? Answer: Machine learning enables computers to learn from data",
            "Question: What is a transformer? Answer: A transformer is a neural network architecture using attention",
            "Question: What is GPT? Answer: GPT is a generative pretrained transformer for text generation",
            "Question: What is attention? Answer: Attention is a mechanism to focus on relevant parts of input",
            // 技术QA
            "Question: What is backpropagation? Answer: Backpropagation is an algorithm for training neural networks",
            "Question: What is gradient descent? Answer: Gradient descent is an optimization algorithm",
            "Question: What is overfitting? Answer: Overfitting is when a model memorizes training data",
            "Question: What is regularization? Answer: Regularization prevents overfitting in models",
            "Question: What is dropout? Answer: Dropout randomly disables neurons during training",
            "Question: What is transfer learning? Answer: Transfer learning reuses knowledge from pretrained models",
            "Question: What is fine tuning? Answer: Fine tuning adapts pretrained models to new tasks",
            "Question: What is tokenization? Answer: Tokenization splits text into smaller units",
            // 应用QA
            "Question: What is image recognition? Answer: Image recognition identifies objects in images",
            "Question: What is speech recognition? Answer: Speech recognition converts audio to text",
            "Question: What is sentiment analysis? Answer: Sentiment analysis detects emotional tone in text",
            "Question: What is machine translation? Answer: Machine translation converts text between languages",
            "Question: What is text classification? Answer: Text classification assigns labels to documents",
            "Question: What is named entity recognition? Answer: NER extracts entities from text",
            // 模型QA
            "Question: What is CNN? Answer: CNN is convolutional neural network for image processing",
            "Question: What is RNN? Answer: RNN is recurrent neural network for sequential data",
            "Question: What is LSTM? Answer: LSTM is long short term memory for learning sequences",
            "Question: What is embedding? Answer: Embedding represents words as dense vectors",
            "Question: What is softmax? Answer: Softmax converts logits to probability distribution",
            "Question: What is loss function? Answer: Loss function measures prediction errors"
    );

    private static
    List<String> finetuneValTexts = Arrays.asList(
            "Question: What is machine learning? Answer: Machine learning enables computers to learn",
            "Question: What is fine tuning? Answer: Fine tuning adapts pretrained models to new tasks",
            "Question: What is tokenization? Answer: Tokenization splits text into smaller units",
            "Question: What is machine learning? Answer: Machine learning enables computers to learn from data",
            "Question: What is embedding? Answer: Embedding represents words as dense vectors"
    );

    // 共享的tokenizer，确保训练和推理使用同一个
    private static GPT1Dataset.SimpleTokenizer sharedTokenizer = new GPT1Dataset.SimpleTokenizer();

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("GPT-1 完整训练与推理演示");
        System.out.println("=".repeat(70));

        try {
            // 演示1: 预训练
            GPT1Model pretrainedModel = demoPretraining();

            // 演示2: 微调(使用预训练模型)
            GPT1Model finetunedModel = demoFinetuning(pretrainedModel);

            // 演示3: 推理
            demoInference(finetunedModel);

            System.out.println("\n" + "=".repeat(70));
            System.out.println("✅ 演示完成!");
            System.out.println("=".repeat(70));
        } catch (Exception e) {
            System.err.println("演示过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 演示1: 预训练流程
     *
     * @return 预训练后的模型
     */
    private static GPT1Model demoPretraining() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("📚 演示1: GPT-1预训练 (Pretrain)");
        System.out.println("=".repeat(70));

        // 1. 准备数据集(先构建词汇表)
        System.out.println("\n📝 步骤1: 准备预训练数据");


        // 先用文本构建词汇表(包含预训练+微调所有数据)
        for (String text : preTrainTexts) {
            sharedTokenizer.encode(text);
        }

        for (String text : finetuneTexts) {
            sharedTokenizer.encode(text);
        }
        int actualVocabSize = sharedTokenizer.getVocabSize();
        System.out.println("✓ 词汇表构建完成");
        System.out.println("  - 词汇表大小: " + actualVocabSize);

        // 2. 创建模型(使用实际词汇表大小)
        System.out.println("\n📝 步骤2: 创建模型");
        GPT1Config config = GPT1Config.createTinyConfig();
        config.setVocabSize(actualVocabSize);  // 设置实际词汇表大小
        GPT1Model model = new GPT1Model("gpt1-pretrain-demo", config);
        System.out.println("✓ 模型创建成功");
        System.out.println("  - 配置: Tiny");
        System.out.println("  - 词汇表大小: " + config.getVocabSize());
        System.out.println("  - 隐藏维度: " + config.getNEmbd());
        System.out.println("  - 层数: " + config.getNLayer());
        System.out.println("  - 注意力头: " + config.getNHead());

        // 3. 加载数据集
        System.out.println("\n📝 步骤3: 加载数据集");
        GPT1Dataset dataset = new GPT1Dataset(
                config.getNPositions(),  // maxSeqLen
                2,                       // batchSize(减小以节省内存)
                actualVocabSize          // vocabSize
        );
        dataset.loadFromTexts(preTrainTexts, sharedTokenizer);
        System.out.println("✓ 数据加载完成");
        System.out.println("  - 样本数: " + dataset.getSampleCount());

        // 4. 配置并开始预训练
        System.out.println("\n📝 步骤4: 开始预训练");
        GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
        trainer.configure(
                5,        // maxEpochs
                5e-3f,    // learningRate
                20,       // warmupSteps
                1.0f      // maxGradNorm
        ).setCheckpoint("./checkpoints/pretrain_demo", 500);

        System.out.println("开始训练...");
        trainer.train();

        System.out.println("\n✅ 预训练完成!");

        return model;
    }

    /**
     * 演示2: 微调流程
     *
     * @param pretrainedModel 预训练的模型
     * @return 微调后的模型
     */
    private static GPT1Model demoFinetuning(GPT1Model pretrainedModel) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("🎯 演示2: GPT-1微调 (Finetune/Posttrain)");
        System.out.println("=".repeat(70));

        // 1. 使用预训练模型
        System.out.println("\n📝 步骤1: 加载预训练模型");
        GPT1Model model = pretrainedModel;
        GPT1Config config = model.getConfig();
        System.out.println("✓ 预训练模型加载完成");
        System.out.println("  - 模型名称: " + model.getName());
        System.out.println("  - 参数量: " + model.getAllParams().size());

        // 2. 准备微调数据
        System.out.println("\n📝 步骤2: 准备微调数据");


        GPT1Dataset trainDataset = new GPT1Dataset(
                config.getNPositions(), 2, sharedTokenizer.getVocabSize() + 10
        );
        trainDataset.loadFromTexts(finetuneTexts, sharedTokenizer);

        GPT1Dataset valDataset = new GPT1Dataset(
                config.getNPositions(), 1, sharedTokenizer.getVocabSize() + 10
        );
        valDataset.loadFromTexts(finetuneValTexts, sharedTokenizer);

        System.out.println("✓ 微调数据准备完成");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());

        // 3. 配置并开始微调
        System.out.println("\n📝 步骤3: 开始微调");
        GPT1Finetune finetuner = new GPT1Finetune(model, trainDataset, valDataset);
        finetuner.configure(
                2,        // maxEpochs
                5e-4f,    // learningRate(比预训练小)
                2         // patience
        ).setCheckpoint("./checkpoints/finetune_demo", 50);

        // 实际执行微调
        System.out.println("开始微调...");
        finetuner.train();

        System.out.println("\n✅ 微调完成!");
        System.out.println("\n📊 微调阶段说明:");
        System.out.println("  - 目标: 适应特定任务");
        System.out.println("  - 数据: 任务相关的标注数据");
        System.out.println("  - 损失: 任务特定损失");
        System.out.println("  - 学习率: 比预训练小");
        System.out.println("  - 技巧: 早停机制防止过拟合");

        return model;
    }

    /**
     * 演示3: 推理流程
     *
     * @param model 训练好的模型
     */
    private static void demoInference(GPT1Model model) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("🚀 演示3: GPT-1推理与文本生成");
        System.out.println("=".repeat(70));

        // 1. 准备推理器
        System.out.println("\n📝 步骤1: 准备推理器");
        GPT1Inference inference = new GPT1Inference(model);
        System.out.println("✓ 推理器准备完成");

        // 2. 准备提示词(使用共享的tokenizer)
        System.out.println("\n📝 步骤2: 准备提示词");
        String promptText = "Deep learning is";
        List<Integer> promptTokens = sharedTokenizer.encode(promptText);
        int[] promptIds = promptTokens.stream().mapToInt(i -> i).toArray();

        System.out.println("✓ 提示文本: \"" + promptText + "\"");
        System.out.println("  - Token序列: " + Arrays.toString(promptIds));
        System.out.println("  - Token数量: " + promptIds.length);
        System.out.println("  - 词汇表大小: " + sharedTokenizer.getVocabSize());

        // 3. 执行实际文本生成
        System.out.println("\n📝 步骤3: 文本生成演示\n");

        // 策略1: 贪婪解码
        System.out.println("策略1: 贪婪解码 (Greedy Decoding)");
        System.out.println("  - 特点: 始终选择概率最高的token");
        System.out.println("  - 优点: 确定性输出,适合需要一致性的任务");
        System.out.println("  - 缺点: 可能陷入重复模式");
        try {
            int[] greedyResult = inference.generateGreedy(promptIds, 10);
            String greedyText = sharedTokenizer.decode(greedyResult);
            System.out.println("  ✓ 生成结果: \"" + greedyText + "\"");
        } catch (Exception e) {
            System.out.println("  ⚠ 生成跳过: " + e.getMessage());
        }

        // 策略2: Temperature采样
        System.out.println("\n策略2: Temperature采样");
        System.out.println("  - 参数: temperature=0.8");
        System.out.println("  - 特点: 控制输出的随机性");
        System.out.println("  - temperature<1: 更确定性");
        System.out.println("  - temperature>1: 更随机性");
        try {
            int[] tempResult = inference.generateWithTemperature(promptIds, 10, 0.8f);
            String tempText = sharedTokenizer.decode(tempResult);
            System.out.println("  ✓ 生成结果: \"" + tempText + "\"");
        } catch (Exception e) {
            System.out.println("  ⚠ 生成跳过: " + e.getMessage());
        }

        // 策略3: Beam Search
        System.out.println("\n策略3: Beam Search");
        System.out.println("  - 参数: beamSize=3");
        System.out.println("  - 特点: 维护多个候选序列,选择全局最优");
        System.out.println("  - 优点: 生成质量高");
        System.out.println("  - 缺点: 计算开销大");
        try {
            int[] beamResult = inference.generateBeamSearch(promptIds, 10, 3);
            String beamText = sharedTokenizer.decode(beamResult);
            System.out.println("  ✓ 生成结果: \"" + beamText + "\"");
        } catch (Exception e) {
            System.out.println("  ⚠ 生成跳过: " + e.getMessage());
        }

        System.out.println("\n💡 推理阶段说明:");
        System.out.println("  - 输入: 提示词token序列");
        System.out.println("  - 输出: 生成的token序列");
        System.out.println("  - 策略选择:");
        System.out.println("    * 需要确定性: 贪婪解码");
        System.out.println("    * 平衡质量与多样性: Temperature采样");
        System.out.println("    * 最高质量: Beam Search");
        System.out.println("    * 创造性任务: 高temperature的采样");
    }

    /**
     * 完整流程演示
     */
    public static void runCompleteWorkflow() {
        System.out.println("=".repeat(70));
        System.out.println("GPT-1 完整训练流程");
        System.out.println("=".repeat(70));

        // 阶段1: 预训练
        System.out.println("\n阶段1: 预训练 (Pretrain)");
        System.out.println("  目标: 学习语言的通用表示");
        System.out.println("  数据: BooksCorpus (7000本书籍)");
        System.out.println("  任务: 因果语言建模 (预测下一个词)");
        System.out.println("  耗时: 约30天 (8个GPU)");

        // 阶段2: 微调
        System.out.println("\n阶段2: 微调 (Finetune/Posttrain)");
        System.out.println("  目标: 适应下游任务");
        System.out.println("  数据: 任务特定数据集");
        System.out.println("  任务: 文本分类/问答/文本蕴含等");
        System.out.println("  耗时: 约3个epoch");

        // 阶段3: 推理
        System.out.println("\n阶段3: 推理 (Inference)");
        System.out.println("  输入: 提示词");
        System.out.println("  处理: 自回归生成");
        System.out.println("  输出: 生成文本");
        System.out.println("  速度: 毫秒级 (CPU推理)");

        System.out.println("\n训练提示:");
        System.out.println("  1. 预训练需要大量计算资源");
        System.out.println("  2. 微调可以在单卡上完成");
        System.out.println("  3. 使用梯度累积可以模拟更大的batch");
        System.out.println("  4. 定期保存检查点防止训练中断");
        System.out.println("  5. 监控验证集损失防止过拟合");
    }
}
