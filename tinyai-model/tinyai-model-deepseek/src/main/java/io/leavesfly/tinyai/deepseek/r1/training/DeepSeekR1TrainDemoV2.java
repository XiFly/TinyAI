package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;

import java.io.*;
import java.util.*;

/**
 * DeepSeek-R1完整训练演示 V2版本
 * 
 * 参考DeepSeekV3TrainDemoV2的实现方式，提供完整的训练流程：
 * 1. 准备真实的教学数据集（适用于教育学习）
 * 2. 预训练阶段 - 基础语言建模训练
 * 3. 后训练阶段 - 任务特定微调
 * 4. 强化学习阶段 - RLHF训练（DeepSeek-R1特色）
 * 5. 推理阶段 - 多种生成策略演示
 * 
 * 改进点：
 * - 使用真实文本数据而非随机数据
 * - 支持从文件加载数据集
 * - 包含数据集自动生成功能
 * - 详细的训练过程说明和日志
 * - 完整的预训练-后训练-强化学习-推理流程
 * 
 * R1特色：
 * - 推理能力增强（Reasoning Enhancement）
 * - 反思机制（Self-Reflection）
 * - 强化学习对齐（RLHF）
 * - 推理过程可视化
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1TrainDemoV2 {
    
    private static SimpleTokenizer sharedTokenizer = new SimpleTokenizer();
    
    private static final String DATA_DIR = "./data/deepseek_r1_training";
    private static final String CHECKPOINT_DIR = "./checkpoints/deepseek_r1_v2";
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 完整训练与推理演示 V2");
        System.out.println("适用于教学和学习的小型数据集训练方案");
        System.out.println("特色：推理增强 + 自我反思 + 强化学习对齐");
        System.out.println("=".repeat(80));
        
        try {
            // 步骤0: 准备数据集文件
            prepareDatasets();
            
            // 步骤1: 预训练（无监督语言建模）
            DeepSeekR1Model pretrainedModel = runPretraining();
            
            // 步骤2: 后训练/微调（有监督学习）
            DeepSeekR1Model finetunedModel = runPosttraining(pretrainedModel);
            
            // 步骤3: 强化学习训练（RLHF - R1核心特色）
            DeepSeekR1Model alignedModel = runRLHFTraining(finetunedModel);
            
            // 步骤4: 推理测试
            runInference(alignedModel);
            
            System.out.println("\n" + "=".repeat(80));
            System.out.println("✅ DeepSeek-R1完整训练流程演示成功!");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("❌ 训练过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // ========== 步骤0: 准备数据集 ==========
    
    /**
     * 准备训练数据集
     * 生成pretrain、posttrain和rlhf数据文件
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
        
        // 生成RLHF强化学习数据集（R1特色）
        generateRLHFDataset();
        
        System.out.println("\n✅ 数据集准备完成!");
    }
    
    /**
     * 生成预训练数据集
     * 包含推理、数学、编程等领域的教学文本
     */
    private static void generatePretrainDataset() throws IOException {
        System.out.println("\n📝 生成预训练数据集...");
        
        List<String> pretrainTexts = new ArrayList<>();
        
        // 1. 推理相关知识 (40条)
        pretrainTexts.addAll(generateReasoningTexts());
        
        // 2. 数学问题解决 (40条)
        pretrainTexts.addAll(generateMathTexts());
        
        // 3. 逻辑推理 (30条)
        pretrainTexts.addAll(generateLogicTexts());
        
        // 4. 编程知识 (30条)
        pretrainTexts.addAll(generateCodingTexts());
        
        // 5. 深度学习基础 (30条)
        pretrainTexts.addAll(generateDeepLearningTexts());
        
        // 6. 反思与自我修正 (30条)
        pretrainTexts.addAll(generateReflectionTexts());
        
        // 写入文件
        String filePath = DATA_DIR + "/pretrain.txt";
        writeToFile(pretrainTexts, filePath);
        
        System.out.println("  ✓ 预训练数据: " + pretrainTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + filePath);
    }
    
    /**
     * 生成后训练数据集
     * 包含推理和反思任务的指令-回答对
     */
    private static void generatePosttrainDataset() throws IOException {
        System.out.println("\n📝 生成后训练数据集...");
        
        List<String> trainTexts = new ArrayList<>();
        List<String> valTexts = new ArrayList<>();
        
        // 训练集: 80条任务感知的指令-回答对
        trainTexts.addAll(generateReasoningQA());
        
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
    }
    
    /**
     * 生成RLHF强化学习数据集
     * 包含推理过程和人类反馈奖励
     */
    private static void generateRLHFDataset() throws IOException {
        System.out.println("\n📝 生成RLHF强化学习数据集...");
        
        List<String> rlhfTexts = new ArrayList<>();
        
        // 生成带奖励标注的推理数据
        rlhfTexts.addAll(generateRLHFReasoningData());
        
        // 写入文件
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        writeToFile(rlhfTexts, rlhfPath);
        System.out.println("  ✓ RLHF训练集: " + rlhfTexts.size() + " 条");
        System.out.println("  ✓ 保存路径: " + rlhfPath);
        System.out.println("  ✓ 数据格式: [REWARD:score] 推理过程");
    }
    
    // ========== 数据生成方法 ==========
    
    /**
     * 生成推理相关文本
     */
    private static List<String> generateReasoningTexts() {
        return Arrays.asList(
            "Reasoning is the process of drawing logical conclusions from available information",
            "Chain of thought prompting improves reasoning by showing intermediate steps",
            "DeepSeek R1 uses deep reasoning to solve complex problems step by step",
            "Reasoning requires breaking down complex problems into simpler sub problems",
            "Multi step reasoning involves connecting multiple logical inferences together",
            "Reasoning confidence indicates how certain the model is about its conclusions",
            "Self verification helps ensure reasoning correctness through multiple checks",
            "Reasoning traces show the complete thought process from question to answer",
            "Deliberate reasoning allocates more computation to harder problems",
            "Reasoning under uncertainty requires probabilistic inference techniques",
            "Analogical reasoning transfers knowledge from familiar to novel situations",
            "Causal reasoning identifies cause and effect relationships between events",
            "Deductive reasoning applies general principles to reach specific conclusions",
            "Inductive reasoning generalizes patterns from specific observations",
            "Abductive reasoning finds the most likely explanation for observations",
            "Critical thinking evaluates arguments for logical consistency and validity",
            "Problem decomposition breaks complex tasks into manageable subtasks",
            "Hypothesis generation creates possible explanations to test against evidence",
            "Evidence evaluation assesses the relevance and reliability of information",
            "Conclusion synthesis combines multiple pieces of evidence into final answer",
            "Reasoning depth refers to the number of inference steps required",
            "Reasoning breadth considers multiple solution paths simultaneously",
            "Metacognition involves thinking about thinking and strategy selection",
            "Reasoning verification checks each step for logical correctness",
            "Error detection identifies mistakes in the reasoning process early"
        );
    }
    
    /**
     * 生成数学相关文本
     */
    private static List<String> generateMathTexts() {
        return Arrays.asList(
            "Mathematics requires systematic reasoning to solve problems correctly",
            "Algebraic manipulation transforms equations while preserving equality",
            "Calculus studies rates of change and accumulation of quantities",
            "Probability measures the likelihood of uncertain events occurring",
            "Statistics extracts meaningful patterns from numerical data",
            "Geometry studies shapes sizes and spatial relationships",
            "Number theory explores properties of integers and their relationships",
            "Linear algebra works with vectors matrices and linear transformations",
            "Mathematical proof establishes truth through logical deduction",
            "Arithmetic operations include addition subtraction multiplication division",
            "Equations express equality between mathematical expressions",
            "Functions map inputs to outputs following defined rules",
            "Optimization finds the best solution among many possibilities",
            "Combinatorics counts and arranges objects following constraints",
            "Set theory provides foundations for modern mathematics",
            "Logic provides the formal basis for mathematical reasoning",
            "Word problems translate real world situations into equations",
            "Mathematical modeling represents real systems with equations",
            "Estimation approximates answers when exact calculation is impractical",
            "Verification checks answers by substituting back into original problem"
        );
    }
    
    /**
     * 生成逻辑推理文本
     */
    private static List<String> generateLogicTexts() {
        return Arrays.asList(
            "Logic is the systematic study of valid inference patterns",
            "Propositional logic deals with statements that are true or false",
            "Predicate logic extends propositional logic with quantifiers",
            "Syllogisms are three part arguments with two premises and conclusion",
            "Modus ponens derives conclusion from conditional and its antecedent",
            "Modus tollens derives negation from conditional and negated consequent",
            "Logical fallacies are errors in reasoning that undermine arguments",
            "Contradiction occurs when statement and its negation are both asserted",
            "Consistency means no contradictions can be derived from premises",
            "Validity means conclusion follows necessarily from premises",
            "Soundness means argument is valid with all true premises",
            "Logical equivalence means two statements have same truth value",
            "Implication connects antecedent to consequent conditionally",
            "Conjunction connects statements with logical and operation",
            "Disjunction connects statements with logical or operation",
            "Negation reverses the truth value of a statement"
        );
    }
    
    /**
     * 生成编程相关文本
     */
    private static List<String> generateCodingTexts() {
        return Arrays.asList(
            "Programming transforms algorithms into executable instructions",
            "Debugging identifies and fixes errors in program logic",
            "Code review improves quality through peer examination",
            "Testing verifies that programs behave as expected",
            "Refactoring improves code structure without changing behavior",
            "Algorithm efficiency measures computational resource usage",
            "Data structures organize information for efficient access",
            "Recursion solves problems by calling function on smaller inputs",
            "Iteration repeats operations using loops until condition met",
            "Abstraction hides complexity behind simple interfaces",
            "Modularity divides programs into independent components",
            "Documentation explains code purpose and usage clearly",
            "Version control tracks changes and enables collaboration",
            "Error handling manages exceptions gracefully",
            "Code optimization improves performance and efficiency"
        );
    }
    
    /**
     * 生成深度学习基础文本
     */
    private static List<String> generateDeepLearningTexts() {
        return Arrays.asList(
            "Deep learning uses neural networks with multiple layers",
            "Backpropagation computes gradients through chain rule",
            "Gradient descent optimizes parameters iteratively",
            "Loss functions measure prediction errors",
            "Activation functions introduce nonlinearity",
            "Transformers use attention for sequence processing",
            "Language models predict next tokens in sequences",
            "Pre training learns general representations from data",
            "Fine tuning adapts models to specific tasks",
            "Reinforcement learning optimizes through rewards",
            "Policy gradient methods update action probabilities",
            "Reward shaping guides learning toward desired behavior",
            "Value functions estimate expected future rewards",
            "Human feedback aligns models with preferences",
            "RLHF combines human feedback with reinforcement learning"
        );
    }
    
    /**
     * 生成反思相关文本
     */
    private static List<String> generateReflectionTexts() {
        return Arrays.asList(
            "Self reflection enables models to evaluate their own outputs",
            "Error correction improves answers through iterative refinement",
            "Confidence estimation indicates reliability of responses",
            "Quality assessment scores outputs on multiple dimensions",
            "Chain of thought shows explicit reasoning process",
            "Self verification checks reasoning for logical errors",
            "Iterative improvement refines answers through multiple passes",
            "Metacognitive monitoring tracks reasoning progress",
            "Self critique identifies weaknesses in generated responses",
            "Reasoning revision updates conclusions based on new insights",
            "Answer comparison evaluates multiple solution approaches",
            "Certainty calibration aligns confidence with accuracy",
            "Reasoning transparency makes thought process visible",
            "Error analysis categorizes and learns from mistakes",
            "Self consistency checks whether multiple paths reach same answer"
        );
    }
    
    /**
     * 生成推理问答对（后训练数据）
     */
    private static List<String> generateReasoningQA() {
        List<String> qa = new ArrayList<>();
        
        // 数学推理问答 (30条)
        qa.add("[MATH] Question: What is 15 plus 27? Let me think step by step. First I add the ones: 5 plus 7 equals 12, carry 1. Then tens: 1 plus 2 plus 1 equals 4. Answer: 42");
        qa.add("[MATH] Question: Calculate 8 times 7. Think: I know 8 times 7 equals 56 because 8 times 5 is 40 and 8 times 2 is 16, so 40 plus 16 is 56. Answer: 56");
        qa.add("[MATH] Question: What is half of 48? Reasoning: To find half I divide by 2. 48 divided by 2 equals 24. Answer: 24");
        qa.add("[MATH] Question: If I have 3 groups of 4 apples, how many total? Think: 3 groups times 4 apples per group equals 3 times 4 which is 12 apples. Answer: 12");
        qa.add("[MATH] Question: What is 100 minus 37? Steps: Subtract ones first: 0 minus 7 needs borrowing, so 10 minus 7 is 3. Tens: 9 minus 3 is 6. Answer: 63");
        qa.add("[MATH] Question: What is 25 times 4? Think: 25 times 4 equals 100 because 25 is one quarter of 100. Answer: 100");
        qa.add("[MATH] Question: Calculate 144 divided by 12. Reasoning: 12 times 12 equals 144, so 144 divided by 12 is 12. Answer: 12");
        qa.add("[MATH] Question: What is 7 plus 8 plus 9? Steps: First 7 plus 8 equals 15, then 15 plus 9 equals 24. Answer: 24");
        qa.add("[MATH] Question: Find 20 percent of 50. Think: 20 percent is 0.2, and 0.2 times 50 equals 10. Answer: 10");
        qa.add("[MATH] Question: What is 81 divided by 9? Reasoning: 9 times 9 equals 81, so the answer is 9. Answer: 9");
        qa.add("[MATH] Question: Calculate 6 squared. Think: 6 squared means 6 times 6 which equals 36. Answer: 36");
        qa.add("[MATH] Question: What is the sum of first 5 natural numbers? Steps: 1 plus 2 plus 3 plus 4 plus 5 equals 15. Answer: 15");
        qa.add("[MATH] Question: Find 3 cubed. Reasoning: 3 cubed means 3 times 3 times 3 equals 27. Answer: 27");
        qa.add("[MATH] Question: What is 45 plus 55? Think: Both add up to 100 since 45 plus 55 equals 100. Answer: 100");
        qa.add("[MATH] Question: Calculate 72 minus 28. Steps: I can think of it as 72 minus 30 plus 2 equals 44. Answer: 44");
        
        // 逻辑推理问答 (25条)
        qa.add("[LOGIC] Question: All cats are animals. Tom is a cat. What can we conclude? Reasoning: If all cats are animals and Tom is a cat, then by syllogism Tom must be an animal. Answer: Tom is an animal");
        qa.add("[LOGIC] Question: If it rains then the ground gets wet. It is raining. What follows? Using modus ponens: Given if P then Q and P is true, Q must be true. Answer: The ground is wet");
        qa.add("[LOGIC] Question: If A implies B and B is false, what about A? Reasoning: By modus tollens if B is false and A implies B, then A must be false. Answer: A is false");
        qa.add("[LOGIC] Question: Some birds can fly. Penguins are birds. Can penguins fly? Think: Some means not all, so we cannot conclude penguins can fly. Answer: Not necessarily, some birds cannot fly");
        qa.add("[LOGIC] Question: No reptiles are warm blooded. Snakes are reptiles. Are snakes warm blooded? Deduction: Since no reptiles are warm blooded and snakes are reptiles, snakes are not warm blooded. Answer: No");
        qa.add("[LOGIC] Question: If all A are B and all B are C, what can we say about A and C? Reasoning: By transitivity, all A must be C. Answer: All A are C");
        qa.add("[LOGIC] Question: Either it is day or it is night. It is not day. What follows? Think: By disjunctive syllogism, if one option is false the other must be true. Answer: It is night");
        qa.add("[LOGIC] Question: If P or Q is true, and P is false, what about Q? Reasoning: Since one of P or Q must be true and P is false, Q must be true. Answer: Q is true");
        qa.add("[LOGIC] Question: All squares are rectangles. Is every rectangle a square? Think: No, the converse is not always true. Some rectangles are not squares. Answer: No");
        qa.add("[LOGIC] Question: If no fish are mammals, and whales are mammals, are whales fish? Deduction: Since no fish are mammals and whales are mammals, whales cannot be fish. Answer: No, whales are not fish");
        qa.add("[LOGIC] Question: Some students like math. Some students like science. Can we conclude some students like both? Reasoning: No, these are independent statements about possibly different students. Answer: Not necessarily");
        qa.add("[LOGIC] Question: If today is Saturday, tomorrow is Sunday. Today is Saturday. What is tomorrow? Using modus ponens directly. Answer: Tomorrow is Sunday");
        qa.add("[LOGIC] Question: All prime numbers greater than 2 are odd. Is 7 odd? Think: 7 is prime and greater than 2, so it must be odd. Answer: Yes, 7 is odd");
        
        // 推理过程问答 (25条)
        qa.add("[REASONING] Question: How do you solve complex problems? Answer: Break them into smaller parts, solve each part, then combine solutions. This is called problem decomposition");
        qa.add("[REASONING] Question: What is chain of thought reasoning? Answer: It means showing step by step thinking process, making each inference explicit before reaching final conclusion");
        qa.add("[REASONING] Question: Why is self verification important? Answer: Self verification catches errors early, improves accuracy, and builds confidence in the final answer");
        qa.add("[REASONING] Question: How does reflection improve reasoning? Answer: Reflection allows reviewing and correcting mistakes, leading to more accurate and reliable conclusions");
        qa.add("[REASONING] Question: What makes a good reasoning trace? Answer: A good trace shows clear steps, logical connections between steps, and explicit justification for each inference");
        qa.add("[REASONING] Question: How to approach an unfamiliar problem? Answer: Identify what is given, what is asked, look for patterns or similar problems, then try systematic approaches");
        qa.add("[REASONING] Question: What is analogical reasoning? Answer: Using similarities between known and unknown cases to infer properties or solutions for the unknown case");
        qa.add("[REASONING] Question: Why show intermediate steps in reasoning? Answer: Intermediate steps make reasoning transparent, easier to verify, and help identify where errors occur");
        qa.add("[REASONING] Question: What is the benefit of multiple solution approaches? Answer: Different approaches can verify each other and increase confidence in the final answer");
        qa.add("[REASONING] Question: How to handle uncertainty in reasoning? Answer: Acknowledge uncertainty, consider multiple possibilities, and use probability or likelihood to guide decisions");
        qa.add("[REASONING] Question: What is backward reasoning? Answer: Starting from the goal and working backward to find what conditions or steps are needed to reach it");
        qa.add("[REASONING] Question: How to avoid reasoning errors? Answer: Check assumptions, verify each step, consider counterexamples, and review the logic chain carefully");
        
        // 编程推理问答 (20条)
        qa.add("[CODING] Question: How to find a bug in code? Answer: First reproduce the error, then trace execution step by step, check variable values, and identify where actual differs from expected");
        qa.add("[CODING] Question: What is the time complexity of binary search? Reasoning: Each step halves the search space, so for n elements we need log n steps. Answer: O of log n");
        qa.add("[CODING] Question: Why use recursion? Answer: Recursion naturally expresses problems that have self similar structure, making code more readable and maintainable");
        qa.add("[CODING] Question: How to optimize slow code? Answer: First profile to find bottlenecks, then apply appropriate optimization like better algorithms or caching");
        qa.add("[CODING] Question: What is the difference between stack and queue? Answer: Stack follows last in first out order while queue follows first in first out order");
        qa.add("[CODING] Question: What is time complexity of linear search? Think: We may need to check all n elements, so it is O of n. Answer: O of n");
        qa.add("[CODING] Question: When to use a hash table? Answer: Use hash tables when you need fast average case lookup, insertion, and deletion operations");
        qa.add("[CODING] Question: What is a linked list advantage over array? Answer: Linked lists allow efficient insertion and deletion without shifting elements");
        qa.add("[CODING] Question: How does merge sort work? Reasoning: Divide array in half, sort each half recursively, then merge sorted halves. Answer: Divide and conquer approach");
        qa.add("[CODING] Question: What is dynamic programming? Answer: Solving problems by breaking into overlapping subproblems and storing results to avoid recomputation");
        qa.add("[CODING] Question: Why use unit tests? Answer: Unit tests verify individual components work correctly, catch bugs early, and enable safe refactoring");
        qa.add("[CODING] Question: What is a race condition? Answer: When program behavior depends on timing of uncontrolled events, leading to unpredictable results");
        
        // 反思问答 (15条)
        qa.add("[REFLECTION] Question: How to verify your answer is correct? Answer: Check each step for errors, try alternative approaches, and verify result satisfies original problem constraints");
        qa.add("[REFLECTION] Question: What to do when reasoning seems wrong? Answer: Stop, review the logic, identify the error, and restart from the correct point with corrected reasoning");
        qa.add("[REFLECTION] Question: How to improve reasoning confidence? Answer: Use multiple approaches, verify intermediate steps, and check that conclusion is consistent with all given information");
        qa.add("[REFLECTION] Question: When should you revise your answer? Answer: Revise when you find logical errors, contradictions with given facts, or when a better solution approach is discovered");
        qa.add("[REFLECTION] Question: How to learn from reasoning mistakes? Answer: Analyze what went wrong, understand why the error occurred, and develop strategies to avoid similar mistakes");
        qa.add("[REFLECTION] Question: What is metacognition? Answer: Thinking about your own thinking process, monitoring understanding, and adjusting strategies as needed");
        qa.add("[REFLECTION] Question: How to know if you fully understand a concept? Answer: Try to explain it simply, apply it to new problems, and identify any gaps in your understanding");
        qa.add("[REFLECTION] Question: Why is doubt useful in reasoning? Answer: Healthy doubt prompts verification, prevents overconfidence, and leads to more robust conclusions");
        qa.add("[REFLECTION] Question: How to identify hidden assumptions? Answer: Question each step, ask what must be true for this to work, and consider alternative interpretations");
        qa.add("[REFLECTION] Question: What is the value of explaining your reasoning? Answer: Explaining forces clarity, reveals gaps in logic, and helps others verify and learn from your approach");
        
        return qa;
    }
    
    /**
     * 生成RLHF强化学习数据
     * 格式: [REWARD:分数] 推理过程文本
     */
    private static List<String> generateRLHFReasoningData() {
        List<String> rlhfData = new ArrayList<>();
        
        // 高奖励的正确推理 (20条, reward 0.8-1.0)
        rlhfData.add("[REWARD:0.95] Question: 5 plus 3. Think: 5 plus 3 equals 8. Verified by counting. Answer: 8. Correct and clear.");
        rlhfData.add("[REWARD:0.90] Question: What is 12 divided by 4? Reasoning: 12 divided by 4 means how many 4s in 12. 4 times 3 is 12. Answer: 3");
        rlhfData.add("[REWARD:0.92] Question: All dogs bark. Rex is a dog. Does Rex bark? Logic: Major premise says all dogs bark. Rex is a dog. Therefore Rex barks. Answer: Yes");
        rlhfData.add("[REWARD:0.88] Question: If today is Monday what is tomorrow? Step 1: Days follow Monday Tuesday order. Step 2: Day after Monday is Tuesday. Answer: Tuesday");
        rlhfData.add("[REWARD:0.93] Question: Which is larger 7 or 5? Compare: 7 is greater than 5 because 7 minus 5 equals 2 which is positive. Answer: 7");
        rlhfData.add("[REWARD:0.91] Question: Half of 10 is what? Calculate: Half means divide by 2. 10 divided by 2 equals 5. Answer: 5");
        rlhfData.add("[REWARD:0.89] Question: 3 times 4 equals? Multiply: 3 groups of 4 is 4 plus 4 plus 4 which equals 12. Answer: 12");
        rlhfData.add("[REWARD:0.94] Question: If A then B and A is true what is B? Apply modus ponens: Given A implies B and A is true, B must be true. Answer: B is true");
        rlhfData.add("[REWARD:0.87] Question: 20 minus 8 is? Subtract: Start with 20, take away 8. 20 minus 8 equals 12. Verify: 12 plus 8 is 20. Answer: 12");
        rlhfData.add("[REWARD:0.96] Question: Is 15 odd or even? Check: Odd numbers are not divisible by 2. 15 divided by 2 is 7.5 which is not integer. Answer: 15 is odd");
        
        // 中等奖励的可接受推理 (15条, reward 0.5-0.7)
        rlhfData.add("[REWARD:0.65] Question: 6 plus 7. Answer: 13. Reasoning was brief but correct. Could show more steps.");
        rlhfData.add("[REWARD:0.60] Question: What is 9 times 2? Answer: 18. Correct answer but no reasoning shown.");
        rlhfData.add("[REWARD:0.70] Question: Is a square a rectangle? Answer: Yes because it has four right angles. Partially correct but missing some details.");
        rlhfData.add("[REWARD:0.55] Question: 100 divided by 5. Answer: 20. Correct but verification would improve confidence.");
        rlhfData.add("[REWARD:0.68] Question: Sum of 4 and 9. Answer: 13. Add ones digit 4 plus 9 is 13. Brief but adequate.");
        rlhfData.add("[REWARD:0.62] Question: Next number after 7? Answer: 8. Counting sequence continues to 8. Simple but correct.");
        rlhfData.add("[REWARD:0.58] Question: Double of 6. Answer: 12. Double means multiply by 2, 6 times 2 is 12.");
        rlhfData.add("[REWARD:0.66] Question: Is 10 greater than 3? Answer: Yes. 10 is clearly larger. Could quantify difference.");
        rlhfData.add("[REWARD:0.72] Question: What comes before 5? Answer: 4. In counting order 4 precedes 5. Correct reasoning.");
        rlhfData.add("[REWARD:0.64] Question: 8 minus 3. Answer: 5. Subtraction gives 5. Could verify by addition.");
        
        // 低奖励的需改进推理 (15条, reward 0.2-0.4)
        rlhfData.add("[REWARD:0.25] Question: 7 plus 8. Answer: 14. Error: 7 plus 8 should be 15 not 14. Arithmetic mistake.");
        rlhfData.add("[REWARD:0.30] Question: All cats are pets. Some pets are dogs. Are all cats dogs? Answer: Yes. Error: Invalid syllogism, conclusion does not follow.");
        rlhfData.add("[REWARD:0.35] Question: 5 times 5. Answer: 20. Error: 5 times 5 is 25 not 20. Calculation wrong.");
        rlhfData.add("[REWARD:0.28] Question: 12 divided by 3. Answer: 3. Error: 12 divided by 3 is 4 not 3. Division error.");
        rlhfData.add("[REWARD:0.40] Question: Is 8 even? Answer: Maybe. Error: Should definitively state 8 is even since 8 divided by 2 is 4.");
        rlhfData.add("[REWARD:0.32] Question: What is 15 minus 7? Answer: 7. Error: 15 minus 7 equals 8 not 7. Arithmetic mistake.");
        rlhfData.add("[REWARD:0.38] Question: If P then Q and Q is true what about P? Answer: P is true. Error: Affirming consequent is a fallacy, we cannot conclude P.");
        rlhfData.add("[REWARD:0.22] Question: 9 plus 4. Answer: 12. Error: 9 plus 4 equals 13 not 12. Off by one error.");
        rlhfData.add("[REWARD:0.35] Question: 6 times 7. Answer: 43. Error: 6 times 7 equals 42 not 43. Multiplication error.");
        rlhfData.add("[REWARD:0.29] Question: Half of 14. Answer: 8. Error: Half of 14 is 7 not 8. Division mistake.");
        
        return rlhfData;
    }
    
    // ========== 文件操作方法 ==========
    
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
        
        List<String> pretrainTexts = readFromFile(pretrainPath);
        List<String> posttrainTrainTexts = readFromFile(posttrainTrainPath);
        List<String> posttrainValTexts = readFromFile(posttrainValPath);
        List<String> rlhfTexts = readFromFile(rlhfPath);
        
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
            String cleanText = removeLabels(text);
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
        
        DeepSeekR1Model model = new DeepSeekR1Model("deepseek-r1-pretrain-v2", config);
        
        System.out.println("  ✓ 模型配置: Tiny (教学专用)");
        System.out.println("  ✓ 词汇表大小: " + config.getVocabSize());
        System.out.println("  ✓ 隐藏维度: " + config.getNEmbd());
        System.out.println("  ✓ 层数: " + config.getNLayer());
        System.out.println("  ✓ 注意力头数: " + config.getNHead());
        System.out.println("  ✓ 最大推理步骤: " + config.getMaxReasoningSteps());
        System.out.println("  ✓ 质量评分维度: " + config.getQualityScoreDim());
        
        // 4. 准备数据集
        System.out.println("\n📝 准备训练数据集...");
        // 使用较短序列长度以加快训练速度
        int seqLength = 64;  // 教学演示用，原始序列长度太长会很慢
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
            10,         // maxEpochs (增加轮次充分收敛)
            2e-3f,      // learningRate (降低学习率避免震荡)
            10,         // warmupSteps
            1.0f        // maxGradNorm
        ).setCheckpoint(CHECKPOINT_DIR + "/pretrain", 200);
        trainer.setLogInterval(50);  // 减少日志输出
        trainer.configureParallel(true, 4);  // 启用并行训练 (4线程)
        
        System.out.println("  ✓ 最大轮次: 10");
        System.out.println("  ✓ 学习率: 2e-3");
        System.out.println("  ✓ Warmup步数: 10");
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
        
        List<String> trainTexts = readFromFile(trainPath);
        List<String> valTexts = readFromFile(valPath);
        
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
            1e-4f,      // learningRate (降低学习率稳定训练)
            2           // patience
        );
        
        System.out.println("  ✓ 最大轮次: 3");
        System.out.println("  ✓ 学习率: 1e-4");
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
     * DeepSeek-R1的核心特色
     */
    private static DeepSeekR1Model runRLHFTraining(DeepSeekR1Model finetunedModel) throws IOException {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🏆 步骤3: DeepSeek-R1 强化学习训练 (RLHF) - R1核心特色");
        System.out.println("=".repeat(80));
        System.out.println("💡 RLHF通过人类反馈优化模型的推理和反思质量");
        System.out.println("💡 这是DeepSeek-R1区别于其他模型的关键技术");
        
        // 1. 加载RLHF数据
        System.out.println("\n📝 加载RLHF训练数据...");
        String rlhfPath = DATA_DIR + "/rlhf_train.txt";
        List<String> rlhfTexts = readFromFile(rlhfPath);
        
        System.out.println("  ✓ RLHF样本: " + rlhfTexts.size() + " 条");
        System.out.println("  ✓ 数据包含: 推理过程 + 人类反馈奖励");
        
        // 2. 准备RLHF数据集
        System.out.println("\n📝 准备RLHF数据集...");
        DeepSeekR1Config config = finetunedModel.getConfig();
        
        DeepSeekR1Dataset rlhfDataset = createRLHFDatasetFromTexts(
            rlhfTexts,
            config.getNPositions(),
            2,  // batch size
            config.getVocabSize()
        );
        
        System.out.println("  ✓ RLHF训练样本: " + rlhfDataset.getSampleCount());
        System.out.println("  ✓ 奖励分布: 0.2-1.0 (正确推理获高奖励)");
        
        // 3. 配置RLHF训练器
        System.out.println("\n📝 配置RLHF训练器...");
        DeepSeekR1RLHFTrainer rlhfTrainer = new DeepSeekR1RLHFTrainer(
            finetunedModel,
            rlhfDataset
        );
        
        rlhfTrainer.configure(
            2,          // maxEpochs
            5e-4f,      // learningRate
            1.0f,       // rewardWeight (奖励权重)
            0.5f        // qualityWeight (质量分数权重)
        );
        
        System.out.println("  ✓ 最大轮次: 2");
        System.out.println("  ✓ 学习率: 5e-4");
        System.out.println("  ✓ 奖励权重: 1.0 (人类反馈)");
        System.out.println("  ✓ 质量权重: 0.5 (模型自评)");
        
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
        System.out.println("  - 技巧: 极小学习率 + 奖励信号引导");
        System.out.println("  - R1特色: 平衡人类反馈与模型自评质量");
        System.out.println("\nℹ️ RLHF关键创新:");
        System.out.println("  - 奖励建模: 学习人类对推理质量的偏好");
        System.out.println("  - 策略优化: 最大化期望奖励同时保持生成多样性");
        System.out.println("  - 自我反思: 模型学会评估并改进自己的推理");
        
        return finetunedModel;
    }
    
    // ========== 步骤4: 推理测试 ==========
    
    /**
     * 执行推理测试
     */
    private static void runInference(DeepSeekR1Model model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 步骤4: DeepSeek-R1 推理与文本生成");
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
            String cleanText = removeLabels(text);
            
            // 编码文本
            List<Integer> tokens = sharedTokenizer.encode(cleanText);
            
            // 转换为数组
            int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
            
            // 截断或填充到maxSeqLength
            int[] paddedSeq = new int[maxSeqLength];
            Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
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
            float reward = extractReward(text);
            String cleanText = removeLabels(text);
            
            // 编码文本
            List<Integer> tokens = sharedTokenizer.encode(cleanText);
            
            // 转换为数组
            int[] sequence = tokens.stream().mapToInt(Integer::intValue).toArray();
            
            // 截断或填充
            int[] paddedSeq = new int[maxSeqLength];
            Arrays.fill(paddedSeq, SimpleTokenizer.PAD_TOKEN_ID);
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
     * 提取奖励值
     */
    private static float extractReward(String text) {
        if (text.startsWith("[REWARD:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 8) {
                try {
                    return Float.parseFloat(text.substring(8, endIdx));
                } catch (NumberFormatException e) {
                    return 0.5f;  // 默认中等奖励
                }
            }
        }
        return 0.5f;
    }
    
    /**
     * 移除标签
     */
    private static String removeLabels(String text) {
        // 移除任务类型标签 [MATH] [LOGIC] [REASONING] [CODING] [REFLECTION]
        // 移除奖励标签 [REWARD:x.xx]
        return text.replaceFirst("^\\[REWARD:[\\d.]+\\]\\s*", "")
                   .replaceFirst("^\\[\\w+\\]\\s*", "");
    }
    
    /**
     * 简单分词器
     */
    static class SimpleTokenizer {
        private final Map<String, Integer> vocab;
        private final Map<Integer, String> reverseVocab;
        private int nextId;
        private boolean frozen;
        
        public static final int PAD_TOKEN_ID = 0;
        
        public SimpleTokenizer() {
            this.vocab = new HashMap<>();
            this.reverseVocab = new HashMap<>();
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
                        // 冻结后使用UNK token (id=1)
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
