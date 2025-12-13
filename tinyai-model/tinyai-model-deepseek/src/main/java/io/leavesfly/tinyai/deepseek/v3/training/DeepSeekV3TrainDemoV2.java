package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.deepseek.v3.TaskType;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-V3完整训练演示 V2版本
 * 
 * 参考GPT1TrainDemoV2的实现方式，提供完整的训练流程：
 * 1. 准备真实的教学数据集（适用于教育学习）
 * 2. 预训练阶段 - 基础语言建模训练
 * 3. 后训练阶段 - 任务特定微调
 * 4. 推理阶段 - 多种生成策略演示
 * 
 * 改进点：
 * - 使用真实文本数据而非随机数据
 * - 支持从文件加载数据集
 * - 包含数据集自动生成功能
 * - 详细的训练过程说明和日志
 * - 完整的预训练-后训练-推理流程
 * 
 * V3特色：
 * - MoE架构的负载均衡训练
 * - 任务感知的数据标注和训练
 * - 代码生成任务的专门支持
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3TrainDemoV2 {
    
    private static SimpleTokenizer sharedTokenizer = new SimpleTokenizer();
    
    private static final String DATA_DIR = "./data/deepseek_v3_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/deepseek_v3_v2";
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("=".repeat(80));
        
        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();
            
            // 步骤1: 预训练
            DeepSeekV3Model pretrainedModel = runPretraining();
            
            // 步骤2: 通用后训练（任务感知微调）
            DeepSeekV3Model finetunedModel = runPosttraining(pretrainedModel);
            
            // 步骤2B (可选): 代码生成专项后训练
            // 说明：此步骤强化MoE专家对代码任务的特化能力
            DeepSeekV3Model codeSpecializedModel = runCodePosttraining(finetunedModel);
            
            // 步骤3: 推理测试
            runInference(codeSpecializedModel);
            
            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ 完整训练流程演示成功!");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 准备训练数据集
     * 生成pretrain和posttrain数据文件
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
        
        // 生成后训练数据集
        generatePosttrainDataset();
        
        System.out.println("\n✅ 数据集准备完成!");
    }
    
    /**
     * 生成预训练数据集
     * 包含深度学习、MoE、Transformer等领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");
        
        List<String> pretrainTexts = new ArrayList<>();
        
        // 1. MoE和专家系统相关知识 (50条)
        pretrainTexts.addAll(generateMoETexts());
        
        // 2. DeepSeek和大模型相关知识 (50条)
        pretrainTexts.addAll(generateDeepSeekTexts());
        
        // 3. 任务感知和推理相关知识 (40条)
        pretrainTexts.addAll(generateReasoningTexts());
        
        // 4. 代码生成和编程相关知识 (40条)
        pretrainTexts.addAll(generateCodingTexts());
        
        // 5. Transformer和注意力机制 (40条)
        pretrainTexts.addAll(generateTransformerTexts());
        
        // 6. 深度学习基础 (30条)
        pretrainTexts.addAll(generateDeepLearningTexts());
        
        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }
    
    /**
     * 生成后训练数据集
     * 包含任务特定的指令-回答对
     */
    private static void generatePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成后训练数据集...");
        
        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();
        
        // 训练集: 100条任务感知的指令-回答对
        trainTexts.addAll(generateTaskAwareQA());
        
        // 验证集: 从训练集中抽取15条
        for (int i = 0; i < 15 && i < trainTexts.size(); i++) {
            valTexts.add(trainTexts.get(i));
        }
        
        // 写入训练集
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        writeToFile(trainTexts, trainPath);
        System.out.println("  ✓ 后训练训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + trainPath);
        
        // 写入验证集
        String valPath = DATA_DIR + "/posttrain_val.txt";
        writeToFile(valTexts, valPath);
        System.out.println("  ✓ 后训练验证集: " + valTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + valPath);
        
        // 生成代码专项训练数据集
        generateCodePosttrainDataset();
    }
    
    /**
     * 生成代码生成专项后训练数据集
     * 纯代码任务数据，用于强化MoE专家特化能力
     */
    private static void generateCodePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成代码专项后训练数据集...");
        
        List<String> codeTrainTexts = new ArrayList<>();
        List<String> codeValTexts = new ArrayList<>();
        
        // 训练集: 60条纯代码任务问答
        codeTrainTexts.addAll(generateCodeQA());
        
        // 验证集: 从训练集中抽取10条
        for (int i = 0; i < 10 && i < codeTrainTexts.size(); i++) {
            codeValTexts.add(codeTrainTexts.get(i));
        }
        
        // 写入训练集
        String codeTrainPath = DATA_DIR + "/code_posttrain_train.txt";
        writeToFile(codeTrainTexts, codeTrainPath);
        System.out.println("  ✓ 代码专项训练集: " + codeTrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + codeTrainPath);
        
        // 写入验证集
        String codeValPath = DATA_DIR + "/code_posttrain_val.txt";
        writeToFile(codeValTexts, codeValPath);
        System.out.println("  ✓ 代码专项验证集: " + codeValTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + codeValPath);
    }
    
    /**
     * 生成MoE相关文本
     */
    private static List<String> generateMoETexts() {
        return Arrays.asList(
            "Mixture of Experts is a neural network architecture that uses multiple expert networks",
            "MoE models route inputs to different expert networks based on learned gating mechanisms",
            "The gating network in MoE decides which experts should process each input",
            "MoE achieves sparse activation by only using a subset of experts for each input",
            "Top K routing selects the K most relevant experts for each input token",
            "Load balancing in MoE ensures all experts are utilized evenly during training",
            "MoE models can scale to billions of parameters while maintaining efficient inference",
            "DeepSeek V3 uses eight expert networks with top two routing strategy",
            "Expert networks in MoE specialize in different aspects of the input distribution",
            "MoE reduces computational cost by activating only a fraction of total parameters",
            "Auxiliary loss in MoE encourages balanced expert utilization",
            "Sparse MoE models achieve better performance than dense models with similar compute",
            "Expert capacity limits the number of tokens each expert can process per batch",
            "MoE routing can be learned jointly with the model during training",
            "Switch Transformer uses one expert per token for extreme sparsity",
            "MoE enables training of very large models on limited hardware resources",
            "Expert parallelism allows MoE layers to scale across multiple devices",
            "Dynamic routing in MoE adapts to different input patterns automatically",
            "MoE models show strong few shot learning capabilities",
            "Load balancing loss prevents expert collapse where some experts are never used",
            "MoE gating uses softmax to produce probability distribution over experts",
            "Hard routing selects top K experts while soft routing uses weighted combination",
            "MoE architecture is particularly effective for multi task learning",
            "Expert dropout can improve MoE model robustness and generalization",
            "MoE models can learn hierarchical specialization across expert networks"
        );
    }
    
    /**
     * 生成DeepSeek相关文本
     */
    private static List<String> generateDeepSeekTexts() {
        return Arrays.asList(
            "DeepSeek is a series of advanced language models with innovative architectures",
            "DeepSeek V3 combines MoE with task aware routing for improved performance",
            "Task aware architecture in DeepSeek adapts to different downstream tasks",
            "DeepSeek models support reasoning coding math and multimodal tasks",
            "DeepSeek V3 achieves state of the art results on code generation benchmarks",
            "The reasoning module in DeepSeek enhances logical inference capabilities",
            "DeepSeek supports ten programming languages including Python Java and C plus plus",
            "Code quality in DeepSeek is evaluated on correctness readability efficiency and style",
            "DeepSeek V3 uses multi head latent attention for efficient processing",
            "Task type classification helps DeepSeek route inputs to specialized experts",
            "DeepSeek models can handle sequences up to thousands of tokens",
            "The architecture of DeepSeek V3 enables twenty five percent parameter activation",
            "DeepSeek V3 training uses load balanced MoE loss for expert utilization",
            "Inference in DeepSeek V3 is four times faster than equivalent dense models",
            "DeepSeek supports both causal language modeling and instruction tuning",
            "The reflection module in DeepSeek R1 enables self correction during reasoning",
            "DeepSeek models show strong performance on mathematical problem solving",
            "Multi task learning in DeepSeek improves generalization across domains",
            "DeepSeek V3 pre training uses large scale diverse text corpora",
            "Post training in DeepSeek fine tunes the model for specific applications",
            "DeepSeek achieves competitive results while using fewer active parameters",
            "The gating network in DeepSeek learns task specific expert selection",
            "DeepSeek models support both English and Chinese languages",
            "Code synthesis in DeepSeek generates functionally correct programs",
            "DeepSeek V3 architecture enables efficient deployment on edge devices"
        );
    }
    
    /**
     * 生成推理相关文本
     */
    private static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning in AI involves drawing logical conclusions from available information",
            "Chain of thought prompting improves reasoning by showing intermediate steps",
            "Task aware models adapt their reasoning strategy based on problem type",
            "Mathematical reasoning requires understanding of numerical relationships and operations",
            "Logical inference applies rules to derive new facts from existing knowledge",
            "Multi step reasoning breaks complex problems into manageable sub problems",
            "Reasoning confidence indicates the model certainty in its conclusions",
            "Commonsense reasoning requires understanding of everyday knowledge and context",
            "Analogical reasoning transfers knowledge from familiar to novel situations",
            "Causal reasoning identifies cause and effect relationships between events",
            "Abstract reasoning manipulates concepts without concrete examples",
            "Deductive reasoning applies general principles to specific cases",
            "Inductive reasoning generalizes from specific observations to broad patterns",
            "Abductive reasoning infers the most likely explanation for observations",
            "Spatial reasoning involves understanding geometric relationships and transformations",
            "Temporal reasoning tracks changes and sequences over time",
            "Reasoning modules can be trained to verify their own conclusions",
            "Self correction in reasoning improves accuracy through iterative refinement",
            "Reasoning traces provide interpretability by showing thought process",
            "Meta reasoning involves thinking about thinking and strategy selection"
        );
    }
    
    /**
     * 生成编程相关文本
     */
    private static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming languages provide formal systems for instructing computers",
            "Python is widely used for machine learning and data science applications",
            "Java is a statically typed object oriented programming language",
            "JavaScript enables interactive web applications in browsers",
            "C plus plus offers low level control with high level abstractions",
            "Code generation models translate natural language to executable programs",
            "Syntax correctness ensures code follows language grammar rules",
            "Code readability makes programs easier to understand and maintain",
            "Algorithm efficiency measures computational complexity and resource usage",
            "Code style guidelines promote consistency across programming projects",
            "Debugging identifies and fixes errors in program logic",
            "Unit testing verifies individual components function correctly",
            "Version control tracks changes and enables collaboration on code",
            "Code refactoring improves structure without changing behavior",
            "Documentation explains code purpose usage and implementation",
            "API design defines interfaces for software components",
            "Error handling manages exceptional conditions gracefully",
            "Code optimization improves performance and resource efficiency",
            "Design patterns provide reusable solutions to common problems",
            "Type systems prevent errors by checking data type compatibility"
        );
    }
    
    /**
     * 生成Transformer相关文本
     */
    private static List<String> generateTransformerTexts() {
        return Arrays.asList(
            "Transformer architecture revolutionized natural language processing",
            "Self attention computes relationships between all positions in parallel",
            "Multi head attention captures different types of dependencies simultaneously",
            "Positional encoding adds sequence order information to token embeddings",
            "Query key value projections enable flexible attention computation",
            "Scaled dot product attention prevents gradient issues with large dimensions",
            "Attention weights show which input positions influence each output",
            "Layer normalization stabilizes training in deep transformer networks",
            "Feed forward networks process each position independently in transformers",
            "Residual connections enable training of very deep transformer models",
            "Transformer decoder uses masked self attention for autoregressive generation",
            "Cross attention connects encoder and decoder in sequence to sequence tasks",
            "Pre training on large corpora gives transformers broad language understanding",
            "Fine tuning adapts pre trained transformers to downstream tasks efficiently",
            "Transformers eliminate recurrence enabling parallel sequence processing",
            "Attention visualization reveals linguistic patterns learned by the model",
            "Sparse attention reduces computational cost for long sequences",
            "Relative position encoding captures position relationships more flexibly",
            "Transformer XL extends context through segment level recurrence",
            "GPT uses decoder only transformers for language generation"
        );
    }
    
    /**
     * 生成深度学习基础文本
     */
    private static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers",
            "Backpropagation computes gradients for training neural networks",
            "Gradient descent optimizes network parameters iteratively",
            "Loss functions measure prediction errors during training",
            "Activation functions introduce non linearity into networks",
            "Batch normalization accelerates training and improves stability",
            "Dropout prevents overfitting by randomly disabling neurons",
            "Learning rate controls optimization step size",
            "Adam optimizer adapts learning rates for each parameter",
            "Regularization techniques prevent models from overfitting",
            "Early stopping monitors validation performance to prevent overfitting",
            "Data augmentation increases training data diversity",
            "Transfer learning reuses knowledge from pre trained models",
            "Neural networks can approximate any continuous function",
            "Deep architectures learn hierarchical feature representations",
            "Convolutional networks excel at processing grid structured data",
            "Recurrent networks handle sequential and temporal data",
            "Attention mechanisms focus on relevant input parts",
            "Skip connections help gradients flow in deep networks",
            "Embedding layers map discrete tokens to continuous vectors"
        );
    }
    
    /**
     * 生成任务感知的问答对（后训练数据）
     */
    private static List<String> generateTaskAwareQA() {
        List<String> qa = new ArrayList<>();
        
        // MoE相关问答 (20条)
        qa.add("[REASONING] Question: What is Mixture of Experts? Answer: Mixture of Experts is an architecture that uses multiple specialized expert networks with a gating mechanism to route inputs efficiently");
        qa.add("[REASONING] Question: How does MoE routing work? Answer: MoE routing uses a gating network to compute scores for each expert and selects the top K experts to process each input token");
        qa.add("[REASONING] Question: Why is load balancing important in MoE? Answer: Load balancing ensures all experts are utilized evenly preventing some experts from being overused while others remain idle");
        qa.add("[REASONING] Question: What is sparse activation? Answer: Sparse activation means only a subset of model parameters are active for each input reducing computational cost significantly");
        qa.add("[MATH] Question: If MoE has 8 experts and uses top 2 routing what is the activation ratio? Answer: The activation ratio is 2 divided by 8 which equals 0.25 or 25 percent");
        
        // DeepSeek相关问答 (20条)
        qa.add("[GENERAL] Question: What is DeepSeek V3? Answer: DeepSeek V3 is an advanced language model combining MoE architecture with task aware routing for efficient and high quality text generation");
        qa.add("[REASONING] Question: How does task aware architecture help? Answer: Task aware architecture adapts model behavior based on task type routing inputs to experts specialized for reasoning coding math or other domains");
        qa.add("[CODING] Question: What languages does DeepSeek support? Answer: DeepSeek supports ten programming languages including Python Java C plus plus JavaScript Go Rust TypeScript Ruby PHP and Swift");
        qa.add("[CODING] Question: How is code quality evaluated? Answer: Code quality is evaluated on four dimensions correctness readability efficiency and adherence to style guidelines");
        qa.add("[GENERAL] Question: What are DeepSeek advantages? Answer: DeepSeek advantages include efficient sparse computation task adaptive routing strong code generation and fast inference speed");
        
        // 编程相关问答 (20条)
        qa.add("[CODING] Question: What is Python used for? Answer: Python is used for machine learning data science web development automation scientific computing and general purpose programming");
        qa.add("[CODING] Question: Explain object oriented programming. Answer: Object oriented programming organizes code into objects that combine data and methods providing encapsulation inheritance and polymorphism");
        qa.add("[CODING] Question: What is algorithm complexity? Answer: Algorithm complexity measures computational resources required typically expressed as time complexity and space complexity using big O notation");
        qa.add("[CODING] Question: Why is code readability important? Answer: Code readability makes programs easier to understand maintain debug and extend by other developers or future self");
        qa.add("[CODING] Question: What are design patterns? Answer: Design patterns are reusable solutions to common software design problems providing tested templates for solving recurring challenges");
        
        // Transformer相关问答 (20条)
        qa.add("[REASONING] Question: What is self attention? Answer: Self attention computes relationships between all positions in a sequence allowing each position to attend to all other positions in parallel");
        qa.add("[REASONING] Question: Why use multi head attention? Answer: Multi head attention allows the model to attend to different representation subspaces simultaneously capturing diverse types of relationships");
        qa.add("[GENERAL] Question: What is positional encoding? Answer: Positional encoding injects information about token positions into embeddings since transformers have no inherent notion of sequence order");
        qa.add("[REASONING] Question: How does attention scaling work? Answer: Attention scores are scaled by the square root of dimension to prevent extremely small gradients when dot products grow large");
        qa.add("[GENERAL] Question: What is the transformer advantage? Answer: Transformers enable parallel processing of sequences eliminate vanishing gradients and capture long range dependencies effectively");
        
        // 深度学习基础问答 (20条)
        qa.add("[REASONING] Question: What is backpropagation? Answer: Backpropagation computes gradients of loss with respect to parameters by applying chain rule backwards through the computational graph");
        qa.add("[MATH] Question: How does gradient descent work? Answer: Gradient descent updates parameters by moving in the direction opposite to the gradient scaled by learning rate to minimize loss");
        qa.add("[REASONING] Question: Why use activation functions? Answer: Activation functions introduce non linearity enabling neural networks to learn complex patterns beyond linear relationships");
        qa.add("[GENERAL] Question: What is overfitting? Answer: Overfitting occurs when a model learns training data too well including noise and fails to generalize to new unseen data");
        qa.add("[REASONING] Question: How does dropout prevent overfitting? Answer: Dropout randomly disables neurons during training forcing the network to learn robust features that do not rely on specific neurons");
        
        return qa;
    }
    
    /**
     * 生成代码专项问答数据（纯CODING任务）
     * 用于强化MoE专家对代码任务的特化能力
     */
    private static List<String> generateCodeQA() {
        List<String> codeQA = new ArrayList<>();
        
        // Python相关问答 (15条)
        codeQA.add("[CODING] Question: How to define a function in Python? Answer: Use def keyword followed by function name parentheses for parameters and colon then indent the function body");
        codeQA.add("[CODING] Question: What is list comprehension in Python? Answer: List comprehension provides concise syntax to create lists using bracket notation with for loop and optional if condition");
        codeQA.add("[CODING] Question: How to handle exceptions in Python? Answer: Use try block for code that may raise exceptions except block to catch and handle specific exception types and finally for cleanup");
        codeQA.add("[CODING] Question: What are Python decorators? Answer: Decorators are functions that modify behavior of other functions using at symbol syntax wrapping the original function with additional functionality");
        codeQA.add("[CODING] Question: How to read files in Python? Answer: Use open function with file path and mode then read content with read or readlines method always close file or use with statement");
        
        // Java相关问答 (15条)
        codeQA.add("[CODING] Question: How to declare a class in Java? Answer: Use public class keyword followed by class name then curly braces containing fields methods and constructors");
        codeQA.add("[CODING] Question: What is inheritance in Java? Answer: Inheritance allows a class to inherit fields and methods from parent class using extends keyword promoting code reuse");
        codeQA.add("[CODING] Question: How to create an interface in Java? Answer: Use interface keyword followed by name declare method signatures without implementation classes implement interface with implements keyword");
        codeQA.add("[CODING] Question: What is Java generics? Answer: Generics enable types to be parameters when defining classes interfaces and methods providing compile time type safety");
        codeQA.add("[CODING] Question: How to handle null in Java? Answer: Check for null before dereferencing use Optional class or annotations like NonNull to prevent null pointer exceptions");
        
        // JavaScript相关问答 (10条)
        codeQA.add("[CODING] Question: What are arrow functions in JavaScript? Answer: Arrow functions provide shorter syntax using arrow notation capture this from enclosing scope unlike regular functions");
        codeQA.add("[CODING] Question: How to handle async operations in JavaScript? Answer: Use promises with then and catch or async await syntax for cleaner asynchronous code handling");
        codeQA.add("[CODING] Question: What is closure in JavaScript? Answer: Closure is a function that remembers variables from its outer scope even after outer function has finished executing");
        codeQA.add("[CODING] Question: How to iterate arrays in JavaScript? Answer: Use for loop forEach map filter or for of loop depending on whether you need transformation filtering or simple iteration");
        codeQA.add("[CODING] Question: What is destructuring in JavaScript? Answer: Destructuring extracts values from arrays or objects into distinct variables using bracket or curly brace syntax");
        
        // C++相关问答 (10条)
        codeQA.add("[CODING] Question: What are pointers in C plus plus? Answer: Pointers store memory addresses enabling dynamic memory allocation direct memory access and efficient data structure implementation");
        codeQA.add("[CODING] Question: How to use templates in C plus plus? Answer: Templates enable generic programming using template keyword with type parameters allowing code to work with different data types");
        codeQA.add("[CODING] Question: What is RAII in C plus plus? Answer: Resource Acquisition Is Initialization ties resource lifetime to object lifetime using constructors and destructors for automatic resource management");
        codeQA.add("[CODING] Question: How to handle memory in C plus plus? Answer: Use new for dynamic allocation delete to free memory or prefer smart pointers like unique ptr and shared ptr");
        codeQA.add("[CODING] Question: What are virtual functions in C plus plus? Answer: Virtual functions enable polymorphism allowing derived classes to override base class methods using virtual keyword");
        
        // 通用编程概念 (10条)
        codeQA.add("[CODING] Question: What is time complexity? Answer: Time complexity measures how algorithm runtime grows with input size using big O notation like O of n or O of n squared");
        codeQA.add("[CODING] Question: How to optimize code performance? Answer: Profile code identify bottlenecks use efficient algorithms and data structures minimize memory allocations and avoid premature optimization");
        codeQA.add("[CODING] Question: What is recursion? Answer: Recursion is when function calls itself to solve problem by breaking it into smaller subproblems requires base case to terminate");
        codeQA.add("[CODING] Question: How to debug code effectively? Answer: Use debugger set breakpoints inspect variables trace execution flow write unit tests and use logging strategically");
        codeQA.add("[CODING] Question: What is code refactoring? Answer: Refactoring improves code structure readability and maintainability without changing external behavior through small incremental changes");
        
        return codeQA;
    }
    
    /**
     * 将文本列表写入文件
     */
    private static void writeToFile(List<String> texts, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String text : texts) {
                writer.write(text);
                writer.newLine();
            }
        }
    }
    
    /**
     * 执行预训练
     */
    private static DeepSeekV3Model runPretraining() throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("📚 步骤1: DeepSeek-V3 预训练 (Pretrain)");
        System.out.println("=".repeat(80));
        
        // 1. 读取所有数据用于构建完整词汇表
        System.out.println("\n📝 加载所有数据以构建词汇表...");
        String pretrainPath = DATA_DIR + "/pretrain.txt";
        String posttrainTrainPath = DATA_DIR + "/posttrain_train.txt";
        String posttrainValPath = DATA_DIR + "/posttrain_val.txt";
        
        List<String> pretrainTexts = readFromFile(pretrainPath);
        List<String> posttrainTrainTexts = readFromFile(posttrainTrainPath);
        List<String> posttrainValTexts = readFromFile(posttrainValPath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练训练数据: " + posttrainTrainTexts.size() + " 条");
        System.out.println("  ✓ 后训练验证数据: " + posttrainValTexts.size() + " 条");
        
        // 2. 基于所有数据构建完整词汇表
        System.out.println("\n📝 构建完整词汇表...");
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(pretrainTexts);
        allTexts.addAll(posttrainTrainTexts);
        allTexts.addAll(posttrainValTexts);
        
        // 遍历所有文本构建词汇表
        for (String text : allTexts) {
            // 移除任务标签后再编码
            String cleanText = removeTaskLabel(text);
            sharedTokenizer.encode(cleanText);
        }
        int vocabSize = sharedTokenizer.getVocabSize();
        
        // 冻结词汇表
        sharedTokenizer.freeze();
        
        System.out.println("  ✓ 完整词汇表大小: " + vocabSize);
        System.out.println("  ✓ 词汇表已冻结,后续不再增加新词");
        
        // 3. 创建DeepSeek-V3模型
        System.out.println("\n📝 创建DeepSeek-V3模型...");
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        config.setVocabSize(vocabSize);
        
        DeepSeekV3Model model = new DeepSeekV3Model("deepseek-v3-pretrain-v2", config);
        
        System.out.println("  ✓ 模型配置: Micro (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getNEmbd());
        System.out.println("  ✓ 层数: " + config.getNLayer());
        System.out.println("  ✓ 注意力头数: " + config.getNHead());
        System.out.println("  ✓ 专家数量: " + config.getNumExperts());
        System.out.println("  ✓ Top-K选择: " + config.getTopK());
        System.out.println("  ✓ 序列长度: " + config.getNPositions());
        
        // 4. 准备数据集
        System.out.println("\n📝 准备训练数据集...");
        DeepSeekV3Dataset dataset = createDatasetFromTexts(
            pretrainTexts,
            config.getNPositions(),
            4,  // batch size
            config.getVocabSize(),
            false  // 预训练不使用任务标签
        );
        
        System.out.println("  ✓ 训练样本: " + dataset.getSampleCount());
        System.out.println("  ✓ 批次大小: 4");
        System.out.println("  ✓ 序列长度: " + config.getNPositions());
        
        // 5. 配置训练器
        System.out.println("\n📝 配置预训练器...");
        DeepSeekV3Pretrain trainer = new DeepSeekV3Pretrain(model, dataset);
        // 修复：增加训练轮次，减少warmupSteps以适应小数据集
        trainer.configure(
            10,         // maxEpochs (增加到10轮，小数据集需要更多轮次)
            5e-4f,      // learningRate (提高学习率)
            10,         // warmupSteps (减少warmup步数，数据少时需快速进入正常训练)
            1.0f        // maxGradNorm
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 100);
        
        System.out.println("  ✓ 最大轮次: 10 (小数据集需要更多轮次)");
        System.out.println("  ✓ 学习率: 5e-4 (提高学习率加速收敛)");
        System.out.println("  ✓ Warmup步数: 10 (减少warmup适应小数据集)");
        System.out.println("  ✓ MoE负载均衡权重: " + config.getLoadBalanceLossWeight());
        
        // 6. 开始训练
        System.out.println("\n📝 开始预训练...");
        System.out.println("-".repeat(80));
        trainer.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 预训练完成!");
        System.out.println("\n💡 预训练阶段总结:");
        System.out.println("  - 目标: 学习语言的通用表示和MoE路由");
        System.out.println("  - 任务: 因果语言建模 + MoE负载均衡");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 特色: 稀疏激活(25%参数) + 专家网络");
        
        return model;
    }
    
    /**
     * 执行后训练/微调
     */
    private static DeepSeekV3Model runPosttraining(DeepSeekV3Model pretrainedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🎯 步骤2: DeepSeek-V3 后训练/微调 (Posttrain)");
        System.out.println("=".repeat(80));
        
        // 1. 加载后训练数据
        System.out.println("\n📝 加载后训练数据...");
        String trainPath = DATA_DIR + "/posttrain_train.txt";
        String valPath = DATA_DIR + "/posttrain_val.txt";
        
        List<String> trainTexts = readFromFile(trainPath);
        List<String> valTexts = readFromFile(valPath);
        
        System.out.println("  ✓ 训练集: " + trainTexts.size() + " 条");
        System.out.println("  ✓ 验证集: " + valTexts.size() + " 条");
        
        // 2. 准备数据集（带任务标签）
        System.out.println("\n📝 准备后训练数据集（任务感知）...");
        DeepSeekV3Config config = pretrainedModel.getConfig();
        
        DeepSeekV3Dataset trainDataset = createDatasetFromTexts(
            trainTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签
        );
        
        DeepSeekV3Dataset valDataset = createDatasetFromTexts(
            valTexts,
            config.getNPositions(),
            1,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签
        );
        
        System.out.println("  ✓ 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + valDataset.getSampleCount());
        System.out.println("  ✓ 任务感知标注: 启用");
        
        // 3. 配置后训练器
        System.out.println("\n📝 配置后训练器...");
        DeepSeekV3Posttrain posttrain = new DeepSeekV3Posttrain(
            pretrainedModel,
            trainDataset,
            valDataset
        );
        
        // 修复：增加训练轮次以适应小数据集
        posttrain.configure(
            5,          // maxEpochs (增加轮次)
            5e-5f,      // learningRate (适当提高)
            3           // patience (增加耐心值)
        );
        
        System.out.println("  ✓ 最大轮次: 5 (小数据集需要更多轮次)");
        System.out.println("  ✓ 学习率: 5e-5 (适当提高)");
        System.out.println("  ✓ 早停耐心值: 3");
        
        // 4. 开始后训练
        System.out.println("\n📝 开始后训练...");
        System.out.println("-".repeat(80));
        posttrain.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 后训练完成!");
        System.out.println("\n💡 后训练阶段总结:");
        System.out.println("  - 目标: 适应任务特定的推理和生成");
        System.out.println("  - 任务: 任务感知的指令跟随");
        System.out.println("  - 数据: 带任务标签的指令数据");
        System.out.println("  - 技巧: 小学习率 + 早停 + 任务路由");
        System.out.println("  - 结果: 模型获得任务感知能力");
        
        return pretrainedModel;
    }
    
    /**
     * 执行代码生成专项后训练
     * 纯代码任务数据，强化MoE专家对代码任务的特化能力
     */
    private static DeepSeekV3Model runCodePosttraining(DeepSeekV3Model finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("💻 步骤2B: DeepSeek-V3 代码生成专项后训练");
        System.out.println("=".repeat(80));
        System.out.println("💡 目标：强化MoE专家对代码任务的特化能力");
        System.out.println("💡 策略：纯代码任务数据 + 更小学习率 + 更多训练轮次");
        
        // 1. 加载代码专项数据
        System.out.println("\n📝 加载代码专项数据...");
        String codeTrainPath = DATA_DIR + "/code_posttrain_train.txt";
        String codeValPath = DATA_DIR + "/code_posttrain_val.txt";
        
        List<String> codeTrainTexts = readFromFile(codeTrainPath);
        List<String> codeValTexts = readFromFile(codeValPath);
        
        System.out.println("  ✓ 代码专项训练集: " + codeTrainTexts.size() + " 条");
        System.out.println("  ✓ 代码专项验证集: " + codeValTexts.size() + " 条");
        System.out.println("  ✓ 任务类型: 纯CODING (所有数据都是代码任务)");
        
        // 2. 准备数据集（纯CODING任务）
        System.out.println("\n📝 准备代码专项数据集...");
        DeepSeekV3Config config = finetunedModel.getConfig();
        
        DeepSeekV3Dataset codeTrainDataset = createDatasetFromTexts(
            codeTrainTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签（全是CODING）
        );
        
        DeepSeekV3Dataset codeValDataset = createDatasetFromTexts(
            codeValTexts,
            config.getNPositions(),
            1,  // batch size
            config.getVocabSize(),
            true  // 使用任务标签
        );
        
        System.out.println("  ✓ 训练样本: " + codeTrainDataset.getSampleCount());
        System.out.println("  ✓ 验证样本: " + codeValDataset.getSampleCount());
        System.out.println("  ✓ 支持语言: Python, Java, JavaScript, C++");
        
        // 3. 配置代码专项后训练器
        System.out.println("\n📝 配置代码专项后训练器...");
        DeepSeekV3Posttrain codePosttrain = new DeepSeekV3Posttrain(
            finetunedModel,
            codeTrainDataset,
            codeValDataset
        );
        
        // 代码任务使用更小的学习率和更多的轮次
        codePosttrain.configure(
            6,          // maxEpochs (代码任务需要更多轮次)
            2e-5f,      // learningRate (适当调整)
            3           // patience
        );
        
        System.out.println("  ✓ 最大轮次: 6 (代码任务需要更多轮次)");
        System.out.println("  ✓ 学习率: 2e-5 (适当调整)");
        System.out.println("  ✓ 早停耐心值: 3");
        System.out.println("  ✓ 专家特化: 持续激活CODING专家");
        
        // 4. 开始代码专项后训练
        System.out.println("\n📝 开始代码生成专项后训练...");
        System.out.println("-".repeat(80));
        codePosttrain.train();
        System.out.println("-".repeat(80));
        
        System.out.println("\n✅ 代码专项后训练完成!");
        System.out.println("\n💡 代码专项后训练阶段总结:");
        System.out.println("  - 目标: 强化MoE专家对代码任务的特化");
        System.out.println("  - 任务: 纯CODING任务的指令跟随");
        System.out.println("  - 数据: 60条代码生成问答 (Python/Java/JS/C++)");
        System.out.println("  - 特色: 持续激活同一批专家 -> 专家特化能力增强");
        System.out.println("  - 结果: 模型获得更强的代码生成能力");
        System.out.println("\nℹ️ MoE专家特化说明:");
        System.out.println("  - 通用后训练: CODING数据仅占~20%, 专家激活模式混合");
        System.out.println("  - 代码专项训练: CODING数据100%, 持续强化特定专家");
        System.out.println("  - 预期效果: Expert 2,5成为代码专家, CODING任务时激活概率大幅提升");
        
        return finetunedModel;
    }
    
    /**
     * 执行推理测试
     */
    private static void runInference(DeepSeekV3Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤3: DeepSeek-V3 推理与文本生成");
        System.out.println("=".repeat(80));
        
        // 1. 创建推理器
        System.out.println("\n📝 创建推理器...");
        DeepSeekV3Inference inference = new DeepSeekV3Inference(model);
        inference.setSeed(42);
        System.out.println("  ✓ 推理器准备完成");
        
        // 2. 测试用例
        TestCase[] testCases = {
            new TestCase("Mixture of Experts is", TaskType.GENERAL),
            new TestCase("DeepSeek V3 combines", TaskType.REASONING),
            new TestCase("Python is used for", TaskType.CODING),
            new TestCase("Self attention computes", TaskType.REASONING)
        };
        
        System.out.println("\n📝 执行文本生成测试...\n");
        
        for (int i = 0; i < testCases.length; i++) {
            TestCase testCase = testCases[i];
            System.out.println("测试 " + (i + 1) + ": \"" + testCase.prompt + "\"");
            System.out.println("任务类型: " + testCase.taskType);
            System.out.println("-".repeat(80));
            
            try {
                List<Integer> tokens = sharedTokenizer.encode(testCase.prompt);
                int[] promptIds = tokens.stream().mapToInt(Integer::intValue).toArray();
                
                // Greedy解码
                System.out.println("  策略1 [Greedy贪婪]: ");
                var greedyResult = inference.generateGreedy(promptIds, 12, testCase.taskType);
                String greedyText = sharedTokenizer.decode(greedyResult.tokens);
                System.out.println("    → " + greedyText);
                
                // Temperature采样
                System.out.println("  策略2 [Temperature=0.8]: ");
                var tempResult = inference.generateWithTemperature(
                    promptIds, 12, 0.8f, testCase.taskType
                );
                String tempText = sharedTokenizer.decode(tempResult.tokens);
                System.out.println("    → " + tempText);
                
                // Top-K采样
                System.out.println("  策略3 [Top-K=50]: ");
                var topKResult = inference.generateTopK(promptIds, 12, 50, testCase.taskType);
                String topKText = sharedTokenizer.decode(topKResult.tokens);
                System.out.println("    → " + topKText);
                
            } catch (Exception e) {
                System.out.println("  ⚠ 生成失败: " + e.getMessage());
            }
            
            System.out.println();
        }
        
        System.out.println("✅ 推理测试完成!");
        System.out.println("\n💡 推理阶段总结:");
        System.out.println("  - 输入: 提示词 + 任务类型");
        System.out.println("  - 处理: 任务感知的自回归生成");
        System.out.println("  - 输出: 生成的完整文本");
        System.out.println("  - 策略: Greedy/Temperature/TopK/TopP");
        System.out.println("  - 特色: MoE稀疏激活 + 任务路由");
    }
    
    /**
     * 从文本创建数据集
     */
    private static DeepSeekV3Dataset createDatasetFromTexts(
            List<String> texts,
            int maxSeqLength,
            int batchSize,
            int vocabSize,
            boolean useTaskLabels) {
        
        List<int[]> sequences = new ArrayList<>();
        List<TaskType> taskTypes = new ArrayList<>();
        
        for (String text : texts) {
            TaskType taskType = TaskType.GENERAL;
            String cleanText = text;
            
            if (useTaskLabels) {
                // 提取任务标签
                taskType = extractTaskType(text);
                cleanText = removeTaskLabel(text);
            }
            
            // 编码文本
            List<Integer> tokens = sharedTokenizer.encode(cleanText);
            
            // 转换为数组
            int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
            
            // 截断或填充到maxSeqLength
            // 显式使用PAD_TOKEN_ID填充，避免与词汇ID冲突
            int[] paddedSeq = new int[maxSeqLength];
            java.util.Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
            int copyLen = Math.min(sequence.length, maxSeqLength);
            System.arraycopy(sequence, 0, paddedSeq, 0, copyLen);
            
            sequences.add(paddedSeq);
            taskTypes.add(taskType);
        }
        
        return new DeepSeekV3Dataset(sequences, taskTypes, maxSeqLength, batchSize, true);
    }
    
    /**
     * 提取任务类型标签
     */
    private static TaskType extractTaskType(String text) {
        if (text.startsWith("[REASONING]")) return TaskType.REASONING;
        if (text.startsWith("[CODING]")) return TaskType.CODING;
        if (text.startsWith("[MATH]")) return TaskType.MATH;
        if (text.startsWith("[MULTIMODAL]")) return TaskType.MULTIMODAL;
        return TaskType.GENERAL;
    }
    
    /**
     * 移除任务标签
     */
    private static String removeTaskLabel(String text) {
        return text.replaceFirst("^\\[\\w+\\]\\s*", "");
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
     * 测试用例类
     */
    private static class TestCase {
        final String prompt;
        final TaskType taskType;
        
        TestCase(String prompt, TaskType taskType) {
            this.prompt = prompt;
            this.taskType = taskType;
        }
    }
    
    /**
     * 简单分词器（类似GPT1的SimpleTokenizer）
     * 注意：id=0保留给PAD token，避免与词汇冲突
     */
    static class SimpleTokenizer {
        private final Map<String, Integer> vocab;
        private final Map<Integer, String> reverseVocab;
        private int nextId;
        private boolean frozen;
        
        /** PAD token的ID，用于填充 */
        public static final int PAD_TOKEN_ID = 0;
        
        public SimpleTokenizer() {
            this.vocab = new HashMap<>();
            this.reverseVocab = new HashMap<>();
            // id=0保留给PAD，词汇从1开始
            this.nextId = 1;
            this.frozen = false;
            // 预注册PAD token
            this.vocab.put("<PAD>", PAD_TOKEN_ID);
            this.reverseVocab.put(PAD_TOKEN_ID, "<PAD>");
        }
        
        public List<Integer> encode(String text) {
            String[] words = text.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", " ")
                .split("\\s+");
            
            List<Integer> tokens = new ArrayList<>();
            for (String word : words) {
                if (word.isEmpty()) continue;
                
                if (!vocab.containsKey(word)) {
                    if (!frozen) {
                        vocab.put(word, nextId);
                        reverseVocab.put(nextId, word);
                        nextId++;
                    } else {
                        // 冻结后使用UNK token (使用id=1作为UNK，避免与PAD冲突)
                        tokens.add(1);
                        continue;
                    }
                }
                tokens.add(vocab.get(word));
            }
            return tokens;
        }
        
        public String decode(int[] tokens) {
            StringBuilder sb = new StringBuilder();
            for (int token : tokens) {
                // 跳过PAD token
                if (token == PAD_TOKEN_ID) continue;
                if (reverseVocab.containsKey(token)) {
                    if (sb.length() > 0) sb.append(" ");
                    sb.append(reverseVocab.get(token));
                }
            }
            return sb.toString();
        }
        
        public int getVocabSize() {
            return nextId;
        }
        
        public void freeze() {
            this.frozen = true;
        }
    }
}
