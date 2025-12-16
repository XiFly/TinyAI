package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.dataset.PretrainDataset;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.util.*;

/**
 * MiniMind 完整训练演示
 * <p>
 * 参考 DeepSeekV3TrainDemoV2 的实现方式，提供完整的训练流程：
 * 1. 准备真实的教学数据集（适用于教育学习）
 * 2. 预训练阶段 - 无监督语言建模训练
 * 3. 微调阶段 - 监督指令微调（SFT）
 * 4. 强化学习阶段 - RLAIF训练
 * 5. 推理阶段 - 多种生成策略演示
 * <p>
 * 数据集特点：
 * - 超小规模，便于快速执行
 * - 内容清晰，适合教学演示
 * - 覆盖完整训练流程
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class MiniMindTrainDemo {

    /** 共享分词器 - 使用标准 MiniMindTokenizer */
    private static MiniMindTokenizer sharedTokenizer;

    private static final String DATA_DIR = "./data/minimind_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/minimind";

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("MiniMind 完整训练与推理演示");
        System.out.println("适用于教学和学习的超小规模数据集训练方案");
        System.out.println("=".repeat(80));

        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();

            // 步骤1: 无监督预训练
            MiniMindModel pretrainedModel = runUnsupervisedPretraining();

            // 步骤2: 监督微调（SFT）
            MiniMindModel finetunedModel = runSupervisedFinetuning(pretrainedModel);

            // 步骤3: 强化学习训练（RLAIF）
            MiniMindModel rlModel = runReinforcementLearningTraining(finetunedModel);

            // 步骤4: 推理测试
            runInference(rlModel);

            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ 完整训练流程演示成功!");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ========== 步骤0: 数据准备 ==========

    /**
     * 准备训练数据集
     */
    private static void prepareDatasets() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📦 步骤0: 准备训练数据集");
        System.out.println("=".repeat(80));

        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
            System.out.println("✓ 创建数据目录: " + DATA_DIR);
        }

        // 生成预训练数据集
        generatePretrainDataset();

        // 生成监督微调数据集
        generateSFTDataset();

        // 生成强化学习数据集
        generateRLDataset();

        System.out.println("\n✅ 数据集准备完成!");
    }

    /**
     * 生成预训练数据集
     * 包含通用语言知识
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");

        List<String> pretrainTexts = new ArrayList<>();

        // 1. 深度学习基础知识 (30条)
        pretrainTexts.addAll(Arrays.asList(
            "Deep learning is a subset of machine learning that uses neural networks",
            "Neural networks consist of interconnected layers of neurons",
            "Backpropagation is the algorithm used to train neural networks",
            "Gradient descent optimizes neural network parameters",
            "Activation functions introduce non-linearity into neural networks",
            "Convolutional neural networks excel at image processing tasks",
            "Recurrent neural networks process sequential data effectively",
            "Transformer architecture revolutionized natural language processing",
            "Attention mechanism allows models to focus on relevant information",
            "Pre-training followed by fine-tuning is a common training strategy",
            "Overfitting occurs when a model memorizes training data",
            "Regularization techniques prevent overfitting in neural networks",
            "Dropout randomly disables neurons during training",
            "Batch normalization stabilizes training of deep networks",
            "Learning rate controls the speed of gradient descent",
            "Adam optimizer adapts learning rates for each parameter",
            "Loss function measures the difference between prediction and truth",
            "Cross-entropy loss is commonly used for classification",
            "Mean squared error is used for regression problems",
            "Early stopping prevents overfitting by monitoring validation loss",
            "Data augmentation increases training data diversity",
            "Transfer learning reuses pre-trained models for new tasks",
            "Embedding layers convert discrete tokens into continuous vectors",
            "Positional encoding adds position information to embeddings",
            "Multi-head attention processes information in parallel",
            "Feedforward networks transform attention outputs",
            "Layer normalization normalizes activations across features",
            "Residual connections help gradients flow through deep networks",
            "Softmax function converts logits to probabilities",
            "Tokenization splits text into meaningful units"
        ));

        // 2. 自然语言处理知识 (30条)
        pretrainTexts.addAll(Arrays.asList(
            "Language models predict the next word in a sequence",
            "Autoregressive models generate text one token at a time",
            "BERT uses bidirectional context for understanding",
            "GPT models use unidirectional context for generation",
            "Fine-tuning adapts pre-trained models to specific tasks",
            "Text classification assigns categories to documents",
            "Named entity recognition identifies entities in text",
            "Sentiment analysis determines emotional tone of text",
            "Machine translation converts text between languages",
            "Question answering systems extract answers from context",
            "Summarization condenses long text into key points",
            "Text generation creates coherent natural language",
            "Perplexity measures language model quality",
            "BLEU score evaluates machine translation quality",
            "Word embeddings capture semantic relationships",
            "Byte-pair encoding handles rare words effectively",
            "Subword tokenization balances vocabulary size and coverage",
            "Masked language modeling is used in BERT pre-training",
            "Causal language modeling is used in GPT pre-training",
            "Few-shot learning enables models to learn from examples",
            "Zero-shot learning performs tasks without specific training",
            "Prompt engineering guides model behavior through input design",
            "In-context learning uses examples within the prompt",
            "Instruction tuning teaches models to follow commands",
            "Reinforcement learning from human feedback aligns models",
            "Temperature controls randomness in text generation",
            "Top-k sampling limits choices to k most probable tokens",
            "Top-p sampling uses cumulative probability threshold",
            "Beam search explores multiple generation paths",
            "Greedy decoding always selects the most probable token"
        ));

        // 3. 机器学习概念 (30条)
        pretrainTexts.addAll(Arrays.asList(
            "Supervised learning uses labeled data for training",
            "Unsupervised learning finds patterns without labels",
            "Reinforcement learning learns through rewards and penalties",
            "Classification predicts discrete categories",
            "Regression predicts continuous values",
            "Clustering groups similar data points together",
            "Dimensionality reduction simplifies high-dimensional data",
            "Feature engineering creates informative input variables",
            "Cross-validation assesses model generalization",
            "Train-test split separates data for training and evaluation",
            "Validation set helps tune hyperparameters",
            "Precision measures positive prediction accuracy",
            "Recall measures coverage of actual positives",
            "F1 score balances precision and recall",
            "Accuracy measures overall prediction correctness",
            "Confusion matrix visualizes classification performance",
            "ROC curve plots true positive versus false positive rates",
            "AUC measures area under ROC curve",
            "Bias-variance tradeoff affects model performance",
            "Ensemble methods combine multiple models",
            "Bagging reduces variance through averaging",
            "Boosting sequentially improves weak learners",
            "Random forest uses ensemble of decision trees",
            "Gradient boosting builds trees to correct errors",
            "Neural architecture search automates model design",
            "Hyperparameter tuning optimizes model configuration",
            "Grid search exhaustively tries parameter combinations",
            "Random search samples parameter space randomly",
            "Bayesian optimization uses probabilistic models",
            "Meta-learning enables learning to learn"
        ));

        // 4. AI伦理与应用 (30条)
        pretrainTexts.addAll(Arrays.asList(
            "Artificial intelligence transforms many industries",
            "AI ethics ensures responsible development",
            "Fairness in AI prevents discrimination",
            "Bias in training data leads to biased models",
            "Transparency makes AI decisions interpretable",
            "Explainable AI helps humans understand model reasoning",
            "Privacy protection is crucial in AI systems",
            "Data security prevents unauthorized access",
            "AI safety ensures systems behave as intended",
            "Robustness makes models resilient to attacks",
            "Adversarial examples fool neural networks",
            "Model interpretability reveals decision factors",
            "Feature importance shows influential variables",
            "Attention visualization reveals focus areas",
            "Counterfactual explanations show decision boundaries",
            "AI applications include healthcare diagnostics",
            "Computer vision enables autonomous vehicles",
            "Natural language processing powers virtual assistants",
            "Recommendation systems personalize user experiences",
            "Fraud detection identifies suspicious transactions",
            "Predictive maintenance prevents equipment failures",
            "Drug discovery accelerates pharmaceutical research",
            "Climate modeling predicts environmental changes",
            "Robotics combines AI with physical systems",
            "Speech recognition converts audio to text",
            "Image generation creates realistic visuals",
            "Style transfer applies artistic styles to images",
            "Anomaly detection identifies unusual patterns",
            "Time series forecasting predicts future values",
            "Knowledge graphs organize structured information"
        ));

        // 5. 编程与软件开发 (30条)
        pretrainTexts.addAll(Arrays.asList(
            "Programming languages enable human-computer communication",
            "Python is popular for machine learning development",
            "Java offers robust object-oriented programming",
            "JavaScript powers interactive web applications",
            "Data structures organize and store information",
            "Algorithms solve computational problems efficiently",
            "Version control tracks code changes over time",
            "Git is widely used for version control",
            "Code review improves software quality",
            "Unit testing verifies individual components",
            "Integration testing checks component interactions",
            "Continuous integration automates testing",
            "Software design patterns solve common problems",
            "Object-oriented programming uses classes and objects",
            "Functional programming emphasizes pure functions",
            "Debugging identifies and fixes code errors",
            "Profiling measures code performance",
            "Optimization improves execution speed",
            "Memory management prevents resource leaks",
            "Exception handling manages runtime errors",
            "API design defines software interfaces",
            "Documentation explains code functionality",
            "Code refactoring improves code structure",
            "Modularity breaks code into manageable pieces",
            "Abstraction hides implementation details",
            "Encapsulation bundles data with methods",
            "Inheritance enables code reuse",
            "Polymorphism allows flexible implementations",
            "Dependency injection improves testability",
            "Clean code principles enhance readability"
        ));

        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);

        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    /**
     * 生成监督微调数据集
     * 包含指令-回答对
     */
    private static void generateSFTDataset() throws IOException {
        System.out.println("\n📝 生成监督微调数据集...");

        List<String> sftTrainTexts = new ArrayList<>();
        List<String> sftValTexts = new ArrayList<>();

        // 训练集: 60条指令-回答对
        sftTrainTexts.addAll(Arrays.asList(
            "Question: What is deep learning? Answer: Deep learning is a subset of machine learning using neural networks with multiple layers",
            "Question: Explain backpropagation Answer: Backpropagation is an algorithm that computes gradients to update neural network weights",
            "Question: What is overfitting? Answer: Overfitting occurs when a model memorizes training data instead of learning general patterns",
            "Question: Define gradient descent Answer: Gradient descent is an optimization algorithm that minimizes loss by updating parameters",
            "Question: What are transformers? Answer: Transformers are neural network architectures using self-attention mechanisms",
            "Question: Explain attention mechanism Answer: Attention allows models to focus on relevant parts of input when processing information",
            "Question: What is fine-tuning? Answer: Fine-tuning adapts pre-trained models to specific tasks with additional training",
            "Question: Define reinforcement learning Answer: Reinforcement learning trains agents through rewards and penalties for actions",
            "Question: What is tokenization? Answer: Tokenization splits text into smaller units like words or subwords for processing",
            "Question: Explain embedding layers Answer: Embedding layers convert discrete tokens into continuous vector representations",
            "Instruction: Write a Python function to add two numbers Answer: def add(a, b): return a + b",
            "Instruction: Create a loop to print numbers 1 to 5 Answer: for i in range(1, 6): print(i)",
            "Instruction: Define a class for a person Answer: class Person: def __init__(self, name): self.name = name",
            "Instruction: Implement binary search Answer: Binary search finds elements in sorted arrays by dividing search space",
            "Instruction: Explain list comprehension Answer: List comprehension creates lists using concise syntax: [x*2 for x in range(10)]",
            "Task: Summarize this concept: Neural networks Answer: Networks of artificial neurons that learn patterns from data",
            "Task: Classify this as positive or negative: I love this product Answer: Positive sentiment",
            "Task: Translate to simple terms: Convolutional neural network Answer: Network specialized for processing grid-like data such as images",
            "Task: Generate a creative name for an AI assistant Answer: MindBot - your intelligent companion",
            "Task: Suggest improvements for code readability Answer: Use meaningful variable names and add comments",
            "Question: How does BERT work? Answer: BERT uses bidirectional transformers to understand context from both directions",
            "Question: What is GPT? Answer: GPT is a generative pre-trained transformer for autoregressive language modeling",
            "Question: Explain cross-entropy loss Answer: Cross-entropy measures difference between predicted and true probability distributions",
            "Question: What is batch normalization? Answer: Batch normalization normalizes layer inputs to stabilize and speed up training",
            "Question: Define learning rate Answer: Learning rate controls step size in gradient descent optimization",
            "Instruction: Sort a list in Python Answer: sorted_list = sorted(my_list) or my_list.sort()",
            "Instruction: Handle exceptions in Python Answer: try: risky_code() except Exception as e: handle_error(e)",
            "Instruction: Read a file in Python Answer: with open('file.txt', 'r') as f: content = f.read()",
            "Instruction: Create a dictionary Answer: my_dict = {'key1': 'value1', 'key2': 'value2'}",
            "Instruction: Use list slicing Answer: first_three = my_list[:3], last_two = my_list[-2:]",
            "Task: Explain AI ethics Answer: AI ethics ensures responsible development considering fairness bias and transparency",
            "Task: Compare supervised and unsupervised learning Answer: Supervised uses labels unsupervised finds patterns without labels",
            "Task: Recommend a machine learning algorithm Answer: For classification try random forest or neural networks",
            "Task: Debug this error: IndexError Answer: Check array bounds and ensure index is within valid range",
            "Task: Optimize slow code Answer: Profile to find bottlenecks use efficient algorithms and data structures",
            "Question: What is transfer learning? Answer: Transfer learning reuses pre-trained models for new related tasks",
            "Question: Explain dropout regularization Answer: Dropout randomly disables neurons during training to prevent overfitting",
            "Question: What is a loss function? Answer: Loss function quantifies difference between model predictions and true values",
            "Question: Define activation functions Answer: Activation functions introduce non-linearity enabling networks to learn complex patterns",
            "Question: What is early stopping? Answer: Early stopping halts training when validation performance stops improving",
            "Instruction: Import libraries in Python Answer: import numpy as np, import pandas as pd",
            "Instruction: Create a virtual environment Answer: python -m venv myenv, source myenv/bin/activate",
            "Instruction: Install packages Answer: pip install package_name",
            "Instruction: Format strings in Python Answer: f'Hello {name}' or 'Hello {}'.format(name)",
            "Instruction: Use lambda functions Answer: square = lambda x: x**2",
            "Task: Improve model accuracy Answer: Try feature engineering data augmentation or ensemble methods",
            "Task: Reduce training time Answer: Use smaller batches GPU acceleration or model pruning",
            "Task: Prevent data leakage Answer: Split data before preprocessing keep test set completely separate",
            "Task: Handle imbalanced data Answer: Use oversampling undersampling or class weights",
            "Task: Validate model performance Answer: Use cross-validation and multiple metrics",
            "Question: What is ensemble learning? Answer: Ensemble learning combines multiple models to improve predictions",
            "Question: Explain feature engineering Answer: Feature engineering creates informative variables from raw data",
            "Question: What is regularization? Answer: Regularization adds penalties to prevent overfitting and improve generalization",
            "Question: Define precision and recall Answer: Precision is accuracy of positive predictions recall is coverage of actual positives",
            "Question: What is the bias-variance tradeoff? Answer: Balancing model complexity to minimize both underfitting and overfitting",
            "Instruction: Use NumPy arrays Answer: import numpy as np, arr = np.array([1, 2, 3])",
            "Instruction: Plot data with Matplotlib Answer: import matplotlib.pyplot as plt, plt.plot(x, y), plt.show()",
            "Instruction: Create pandas DataFrame Answer: import pandas as pd, df = pd.DataFrame(data)",
            "Instruction: Apply function to DataFrame Answer: df['new_col'] = df['col'].apply(lambda x: x*2)",
            "Instruction: Split train-test data Answer: from sklearn.model_selection import train_test_split"
        ));

        // 验证集: 从训练集中抽取10条
        for (int i = 0; i < 10 && i < sftTrainTexts.size(); i++) {
            sftValTexts.add(sftTrainTexts.get(i));
        }

        // 写入训练集
        String trainPath = DATA_DIR + "/sft_train.txt";
        writeToFile(sftTrainTexts, trainPath);
        System.out.println("  ✓ SFT训练集: " + sftTrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);

        // 写入验证集
        String valPath = DATA_DIR + "/sft_val.txt";
        writeToFile(sftValTexts, valPath);
        System.out.println("  ✓ SFT验证集: " + sftValTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
    }

    /**
     * 生成强化学习数据集
     * 包含带奖励的样本
     */
    private static void generateRLDataset() throws IOException {
        System.out.println("\n📝 生成强化学习数据集...");

        List<String> rlTexts = new ArrayList<>();

        // 40条带奖励标签的样本
        rlTexts.addAll(Arrays.asList(
            "[REWARD:1.0] Question: What is machine learning? Answer: Machine learning enables computers to learn from data without explicit programming",
            "[REWARD:0.9] Question: Explain neural networks Answer: Neural networks are computing systems inspired by biological brains",
            "[REWARD:0.8] Question: What is deep learning? Answer: Deep learning uses multi-layer neural networks for complex pattern recognition",
            "[REWARD:1.0] Instruction: Write clean code Answer: Use meaningful names add comments and follow style guidelines",
            "[REWARD:0.9] Instruction: Debug efficiently Answer: Use print statements debuggers and unit tests",
            "[REWARD:0.7] Task: Improve performance Answer: Optimize algorithms and use better data structures",
            "[REWARD:0.8] Task: Ensure code quality Answer: Write tests review code and refactor regularly",
            "[REWARD:1.0] Question: What is AI safety? Answer: AI safety ensures systems behave reliably and aligned with human values",
            "[REWARD:0.9] Question: Define model interpretability Answer: Interpretability makes model decisions understandable to humans",
            "[REWARD:0.8] Question: What is fairness in AI? Answer: Fairness prevents discrimination and ensures equitable treatment",
            "[REWARD:1.0] Instruction: Handle errors gracefully Answer: Use try-except blocks and provide informative error messages",
            "[REWARD:0.9] Instruction: Write efficient code Answer: Avoid unnecessary loops and use vectorized operations",
            "[REWARD:0.8] Task: Document your code Answer: Write clear docstrings and maintain README files",
            "[REWARD:0.7] Task: Test thoroughly Answer: Cover edge cases and use both unit and integration tests",
            "[REWARD:1.0] Question: What is gradient descent? Answer: Gradient descent iteratively updates parameters to minimize loss",
            "[REWARD:0.9] Question: Explain overfitting prevention Answer: Use regularization dropout and cross-validation",
            "[REWARD:0.8] Question: What is transfer learning? Answer: Transfer learning applies knowledge from one task to another",
            "[REWARD:1.0] Instruction: Optimize hyperparameters Answer: Use grid search random search or Bayesian optimization",
            "[REWARD:0.9] Instruction: Prevent data leakage Answer: Split data properly and avoid using test information",
            "[REWARD:0.8] Task: Improve model robustness Answer: Use data augmentation and adversarial training",
            "[REWARD:0.7] Task: Monitor model performance Answer: Track metrics and set up alerts for degradation",
            "[REWARD:1.0] Question: What is attention mechanism? Answer: Attention helps models focus on relevant input parts",
            "[REWARD:0.9] Question: Explain transformer architecture Answer: Transformers use self-attention for parallel processing",
            "[REWARD:0.8] Question: What is BERT? Answer: BERT uses bidirectional transformers for language understanding",
            "[REWARD:1.0] Instruction: Design scalable systems Answer: Use modular architecture and efficient algorithms",
            "[REWARD:0.9] Instruction: Ensure reproducibility Answer: Set random seeds and document all parameters",
            "[REWARD:0.8] Task: Validate assumptions Answer: Check data distributions and verify preprocessing steps",
            "[REWARD:0.7] Task: Communicate results Answer: Use visualizations and explain in simple terms",
            "[REWARD:1.0] Question: What is fine-tuning? Answer: Fine-tuning adapts pre-trained models to specific tasks",
            "[REWARD:0.9] Question: Explain data augmentation Answer: Data augmentation increases diversity by transforming existing data",
            "[REWARD:0.8] Question: What is batch normalization? Answer: Batch normalization normalizes inputs to stabilize training",
            "[REWARD:1.0] Instruction: Write modular code Answer: Break complex functions into smaller reusable components",
            "[REWARD:0.9] Instruction: Follow best practices Answer: Use version control write tests and review code",
            "[REWARD:0.8] Task: Optimize memory usage Answer: Use generators avoid copying and release resources",
            "[REWARD:0.7] Task: Profile code performance Answer: Identify bottlenecks and optimize critical paths",
            "[REWARD:1.0] Question: What is ensemble learning? Answer: Ensemble learning combines multiple models for better predictions",
            "[REWARD:0.9] Question: Explain cross-validation Answer: Cross-validation assesses model performance on multiple data splits",
            "[REWARD:0.8] Question: What is feature engineering? Answer: Feature engineering creates informative variables from raw data",
            "[REWARD:1.0] Instruction: Handle edge cases Answer: Test boundary conditions and null inputs",
            "[REWARD:0.9] Task: Maintain code quality Answer: Refactor regularly and eliminate technical debt"
        ));

        // 写入文件
        String filePath = DATA_DIR + "/rl_train.txt";
        writeToFile(rlTexts, filePath);

        System.out.println("  ✓ RL训练数据: " + rlTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }

    // ========== 步骤1: 无监督预训练 ==========

    /**
     * 执行无监督预训练 - 使用标准 PretrainTrainer
     */
    private static MiniMindModel runUnsupervisedPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: MiniMind 无监督预训练 (Unsupervised Pretraining)");
        System.out.println("=".repeat(80));

        // 1. 创建字符级分词器（用于教学演示）
        System.out.println("\n📝 创建分词器...");
        int vocabSize = 1024;  // 足够覆盖教学数据集
        int maxSeqLen = 64;    // 序列长度要足够容纳训练样本
        sharedTokenizer = MiniMindTokenizer.createCharLevelTokenizer(vocabSize, maxSeqLen);
        System.out.println("  ✓ 分词器类型: 字符级 (Char-Level)");
        System.out.println("  ✓ 词汇表大小: " + sharedTokenizer.getVocabulary().getVocabSize());

        // 2. 创建MiniMind模型（超小配置）
        System.out.println("\n📝 创建MiniMind模型...");
        MiniMindConfig config = createMicroConfig(sharedTokenizer.getVocabulary().getVocabSize());
        MiniMindModel model = new MiniMindModel("minimind-pretrain", config);

        System.out.println("  ✓ 模型配置: Micro (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getHiddenSize());
        System.out.println("  ✓ 层数: " + config.getNumLayers());
        System.out.println("  ✓ 注意力头数: " + config.getNumHeads());
        System.out.println("  ✓ 最大序列长度: " + config.getMaxSeqLen());

        // 3. 使用标准 PretrainDataset 加载数据
        System.out.println("\n📝 准备预训练数据集...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        List<String> pretrainTexts = readFromFile(pretrainPath);
        
        int batchSize = 2;  // 小批次便于教学
        PretrainDataset dataset = new PretrainDataset(sharedTokenizer, maxSeqLen, batchSize);
        dataset.loadFromTexts(pretrainTexts);
        dataset.prepare(true);
        System.out.println("  ✓ 预训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 4. 使用标准 PretrainTrainer 进行训练
        System.out.println("\n📝 开始无监督预训练...");
        System.out.println("  - 训练目标: 因果语言建模 (下一个词预测)");
        System.out.println("  - 学习率: 1e-2");
        System.out.println("  - 训练轮次: 3 epochs");
        System.out.println("-".repeat(80));

        PretrainTrainer trainer = new PretrainTrainer(model, dataset);
        trainer.configure(3, 1e-2f, 0, 1.0f);  // 3 epochs, lr=1e-2, no warmup
        trainer.setLogInterval(10);  // 每10步打印一次
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 无监督预训练完成!");
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和语法");
        System.out.println("  - 任务: 因果语言建模（预测下一个词）");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 技巧: 较高学习率 + 多轮训练");

        return model;
    }

    // ========== 步骤2: 监督微调 ==========

    /**
     * 执行监督微调（SFT）- 使用标准 SFTTrainer
     */
    private static MiniMindModel runSupervisedFinetuning(MiniMindModel pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: MiniMind 监督微调 (Supervised Fine-tuning)");
        System.out.println("=".repeat(80));

        // 1. 加载SFT数据
        System.out.println("\n📝 加载监督微调数据...");
        String trainPath = DATA_DIR + "/sft_train.txt";
        List<String> trainTexts = readFromFile(trainPath);
        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");

        // 2. 使用标准 SFTDataset
        System.out.println("\n📝 准备监督微调数据集...");
        MiniMindConfig config = pretrainedModel.getConfig();
        int batchSize = 2;
        
        SFTDataset dataset = new SFTDataset(sharedTokenizer, config.getMaxSeqLen(), batchSize);
        // 将纯文本转换为指令格式
        for (String text : trainTexts) {
            dataset.addSample(text, "", text);  // 简化：指令=输出
        }
        dataset.prepare(true);
        System.out.println("  ✓ 训练样本数: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次数量: " + dataset.getBatchCount());

        // 3. 使用标准 SFTTrainer
        System.out.println("\n📝 开始监督微调训练...");
        System.out.println("  - 训练目标: 指令跟随和对话生成");
        System.out.println("  - 学习率: 1e-3 (比预训练低10倍)");
        System.out.println("  - 训练轮次: 3 epochs");
        System.out.println("-".repeat(80));

        SFTTrainer trainer = new SFTTrainer(pretrainedModel, dataset);
        trainer.configure(3, 1e-3f, 1.0f);  // 3 epochs, lr=1e-3
        trainer.train();

        System.out.println("-".repeat(80));
        System.out.println("\n✅ 监督微调完成!");
        System.out.println("\n💡 SFT阶段总结:");
        System.out.println("  - 目标: 学习遵循指令和生成高质量回答");
        System.out.println("  - 任务: 指令微调（问答对）");
        System.out.println("  - 数据: 带标签的指令-回答数据");
        System.out.println("  - 技巧: 小学习率 + 早停防止过拟合");

        return pretrainedModel;
    }

    // ========== 步骤3: 强化学习训练 ==========

    /**
     * 执行强化学习训练（RLAIF）- 使用简化的奖励加权策略梯度
     * 
     * 核心思想：将奖励作为损失的权重，高奖励样本获得更大的梯度贡献
     * Loss = -reward * log P(y|x)
     */
    private static MiniMindModel runReinforcementLearningTraining(MiniMindModel finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤3: MiniMind 强化学习训练 (Reinforcement Learning)");
        System.out.println("=".repeat(80));
        System.out.println("💡 使用奖励加权的策略梯度方法优化模型");

        // 1. 加载RL数据
        System.out.println("\n📝 加载强化学习训练数据...");
        String rlPath = DATA_DIR + "/rl_train.txt";
        List<String> rlTexts = readFromFile(rlPath);
        System.out.println("  ✓ RL训练数据: " + rlTexts.size() + " 条");

        // 2. 解析数据并提取奖励
        System.out.println("\n📝 准备强化学习数据集...");
        List<String> texts = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        
        for (String line : rlTexts) {
            float reward = extractReward(line);
            String cleanText = removeRewardLabel(line);
            texts.add(cleanText);
            rewards.add(reward);
        }
        
        float avgReward = (float) rewards.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
        System.out.println("  ✓ RL样本数: " + texts.size());
        System.out.println("  ✓ 平均奖励: " + String.format("%.2f", avgReward));

        // 3. 配置训练
        MiniMindConfig config = finetunedModel.getConfig();
        float learningRate = 5e-4f;
        int epochs = 2;
        int logInterval = 10;
        
        System.out.println("\n📝 开始强化学习训练...");
        System.out.println("  - 训练目标: 最大化奖励加权的对数概率");
        System.out.println("  - 算法: 奖励加权策略梯度 (Reward-Weighted Policy Gradient)");
        System.out.println("  - 学习率: " + learningRate);
        System.out.println("  - 训练轮次: " + epochs);
        System.out.println("-".repeat(80));

        // 4. 创建优化器和损失函数
        Adam optimizer = new Adam(finetunedModel, learningRate, 0.9f, 0.999f, 1e-8f);
        SoftmaxCrossEntropy lossFunction = new SoftmaxCrossEntropy();
        finetunedModel.setTraining(true);
        
        int step = 0;
        int maxSeqLen = config.getMaxSeqLen();
        
        // 5. 训练循环
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0.0f;
            int sampleCount = 0;
            
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                float reward = rewards.get(i);
                
                // 编码文本
                List<Integer> tokenIds = sharedTokenizer.encode(text, true, true);
                if (tokenIds.size() < 2) continue;
                
                // 准备输入和目标
                int seqLen = Math.min(tokenIds.size() - 1, maxSeqLen - 1);
                float[] inputData = new float[seqLen];
                float[] targetData = new float[seqLen];
                
                for (int j = 0; j < seqLen; j++) {
                    inputData[j] = tokenIds.get(j);
                    targetData[j] = tokenIds.get(j + 1);
                }
                
                Variable input = new Variable(NdArray.of(inputData, Shape.of(1, seqLen)));
                Variable target = new Variable(NdArray.of(targetData, Shape.of(1, seqLen)));
                
                // 前向传播
                Variable logits = finetunedModel.predict(input);
                
                // 计算损失 (reshape为2D)
                int[] logitsShape = logits.getValue().getShape().getShapeDims();
                int totalTokens = logitsShape[0] * logitsShape[1];
                int vocabSize = logitsShape[2];
                
                Variable logitsReshaped = logits.reshape(Shape.of(totalTokens, vocabSize));
                Variable targetReshaped = target.reshape(Shape.of(totalTokens, 1));
                
                Variable loss = lossFunction.loss(targetReshaped, logitsReshaped);
                
                // 奖励加权：高奖励样本获得更大权重
                Variable weightedLoss = loss.mul(new Variable(NdArray.of(reward)));
                
                // 反向传播
                finetunedModel.clearGrads();
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
        System.out.println("\n💡 RL阶段总结:");
        System.out.println("  - 目标: 通过奖励信号对齐模型行为");
        System.out.println("  - 方法: 奖励加权的交叉熵损失");
        System.out.println("  - 效果: 高奖励样本获得更大梯度贡献");
        System.out.println("  - 技巧: 小学习率 + 奖励引导");

        return finetunedModel;
    }

    // ========== 步骤4: 推理测试 ==========

    /**
     * 执行推理测试
     */
    private static void runInference(MiniMindModel model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤4: MiniMind 推理测试");
        System.out.println("=".repeat(80));

        // 设置为推理模式
        model.setTraining(false);

        // 测试用例
        List<String> testPrompts = Arrays.asList(
            "Question: What is machine learning?",
            "Instruction: Write a Python function",
            "Task: Explain neural networks",
            "Question: Define deep learning"
        );

        System.out.println("\n📝 测试不同生成策略...\n");

        for (String prompt : testPrompts) {
            System.out.println("提示词: " + prompt);

            try {
                // 编码提示词
                List<Integer> promptTokens = sharedTokenizer.encode(prompt);
                int[] promptIds = promptTokens.stream().mapToInt(Integer::intValue).toArray();

                // Greedy解码
                int[] greedyResult = model.generate(
                    promptIds,
                    20,      // maxNewTokens
                    0.0f,    // temperature (greedy)
                    0,       // topK
                    0.0f     // topP
                );

                String greedyText = sharedTokenizer.decode(intArrayToList(greedyResult));
                System.out.println("  [Greedy] → " + greedyText);

            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }

            System.out.println();
        }

        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词文本");
        System.out.println("  - 处理: 自回归生成");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/Top-K/Top-P");
    }

    // ========== 辅助方法 ==========

    /**
     * 创建超小型配置（用于快速演示）
     */
    private static MiniMindConfig createMicroConfig(int vocabSize) {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLen(64);          // 序列长度
        config.setHiddenSize(128);        // 隐藏维度
        config.setNumLayers(2);           // 层数
        config.setNumHeads(4);            // 注意力头数
        config.setFfnHiddenSize(256);     // FFN隐藏维度
        config.setDropout(0.1f);
        config.setEpsilon(1e-5f);
        return config;
    }

    /**
     * int[] 转 List<Integer> 辅助方法
     */
    private static List<Integer> intArrayToList(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int value : array) {
            list.add(value);
        }
        return list;
    }

    /**
     * 提取奖励值
     */
    private static float extractReward(String text) {
        if (text.startsWith("[REWARD:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 0) {
                String rewardStr = text.substring(8, endIdx);
                try {
                    return Float.parseFloat(rewardStr);
                } catch (NumberFormatException e) {
                    return 0.5f;
                }
            }
        }
        return 0.5f;
    }

    /**
     * 移除奖励标签
     */
    private static String removeRewardLabel(String text) {
        return text.replaceFirst("^\\[REWARD:[0-9.]+\\]\\s*", "");
    }

    /**
     * 从文件读取文本
     */
    private static List<String> readFromFile(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }

    /**
     * 写入文件
     */
    private static void writeToFile(List<String> lines, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        }
    }
}
