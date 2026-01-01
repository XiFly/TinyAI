package io.leavesfly.tinyai.deepseek.rx;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeekR1Model单元测试
 * 
 * 测试范围：
 * 1. 模型创建（工厂方法）
 * 2. 前向传播（基础predict）
 * 3. 多步推理功能
 * 4. 自我反思评估
 * 5. 序列生成
 * 6. 模型信息
 * 
 * @author leavesfly
 */
public class DeepSeekR1ModelTest {
    
    private DeepSeekR1Config tinyConfig;
    
    @BeforeEach
    public void setUp() {
        // 使用Tiny配置加快测试速度
        tinyConfig = DeepSeekR1Config.createTinyConfig();
    }
    
    @Test
    public void testCreateStandardModel() {
        // 测试创建标准模型
        DeepSeekR1Model model = DeepSeekR1Model.createStandardModel("test-standard");
        
        assertNotNull(model, "模型应创建成功");
        assertEquals("test-standard", model.getName(), "模型名称应匹配");
        String summary = model.getConfigSummary();
        assertNotNull(summary, "配置摘要应存在");
        assertTrue(summary.contains("DeepSeek-R1"), "摘要应包含DeepSeek-R1");
    }
    
    @Test
    public void testCreateTinyModel() {
        // 测试创建微型模型
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("test-tiny");
        
        assertNotNull(model, "微型模型应创建成功");
        assertEquals("test-tiny", model.getName(), "模型名称应匹配");
    }
    
    @Test
    public void testCreateSmallModel() {
        // 测试创建小型模型
        DeepSeekR1Model model = DeepSeekR1Model.createSmallModel("test-small");
        
        assertNotNull(model, "小型模型应创建成功");
        assertEquals("test-small", model.getName(), "模型名称应匹配");
    }
    
    @Test
    public void testBasicPredict() {
        // 测试基础预测功能
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        // 创建输入: [batch=2, seq=4]
        float[][] inputData = {
            {1, 15, 23, 42},
            {2, 16, 24, 43}
        };
        Variable input = new Variable(NdArray.of(inputData));
        
        // 执行预测
        Variable output = model.predict(input);
        
        assertNotNull(output, "输出应存在");
        assertNotNull(output.getValue(), "输出值应存在");
        
        // 验证输出形状: [batch, seq, vocab_size]
        NdArray outputArray = output.getValue();
        assertEquals(3, outputArray.getShape().getDimNum(), 
                    "输出应为3维张量");
        assertEquals(2, outputArray.getShape().getDimension(0), 
                    "batch维度应为2");
        assertEquals(4, outputArray.getShape().getDimension(1), 
                    "序列维度应为4");
        assertEquals(tinyConfig.getVocabSize(), 
                    outputArray.getShape().getDimension(2), 
                    "词汇表维度应匹配");
    }
    
    @Test
    public void testPredictWithDifferentBatchSizes() {
        // 测试不同batch大小
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        // Batch size = 1
        float[][] input1 = {{1, 2, 3}};
        Variable output1 = model.predict(new Variable(NdArray.of(input1)));
        assertEquals(1, output1.getValue().getShape().getDimension(0), 
                    "batch=1的输出应正确");
        
        // Batch size = 4
        float[][] input4 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        };
        Variable output4 = model.predict(new Variable(NdArray.of(input4)));
        assertEquals(4, output4.getValue().getShape().getDimension(0), 
                    "batch=4的输出应正确");
    }
    
    @Test
    public void testPredictWithDetails() {
        // 测试带详细信息的预测
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        float[][] inputData = {{1, 2, 3, 4, 5}};
        Variable input = new Variable(NdArray.of(inputData));
        
        DeepSeekR1Block.DetailedForwardResult result = 
            model.predictWithDetails(input);
        
        assertNotNull(result, "详细结果应存在");
        assertNotNull(result.logits, "logits应存在");
        assertNotNull(result.reasoningResult, "推理结果应存在");
        assertNotNull(result.reflectionResult, "反思结果应存在");
    }
    
    @Test
    public void testPerformReasoning() {
        // 测试多步推理功能
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        float[][] reasoningPrompt = {{1, 5, 10, 15, 20}};
        Variable input = new Variable(NdArray.of(reasoningPrompt));
        
        DeepSeekR1Model.ReasoningOutput result = model.performReasoning(input);
        
        assertNotNull(result, "推理结果应存在");
        assertNotNull(result.logits, "logits应存在");
        
        // 验证推理步骤数在合理范围内
        assertTrue(result.numSteps > 0 && 
                  result.numSteps <= tinyConfig.getMaxReasoningSteps(),
                  "推理步骤数应在[1, " + tinyConfig.getMaxReasoningSteps() + "]范围内，" +
                  "实际: " + result.numSteps);
        
        // 验证置信度在[0,1]范围内
        assertTrue(result.averageConfidence >= 0 && result.averageConfidence <= 1,
                  "平均置信度应在[0,1]范围内，实际: " + result.averageConfidence);
        
        // 验证质量评分对象存在
        assertNotNull(result.qualityScore, "质量评分对象应存在");
        
        System.out.println("推理步骤数: " + result.numSteps);
        System.out.println("平均置信度: " + result.averageConfidence);
        System.out.println("质量评分: " + result.qualityScore);
    }
    
    @Test
    public void testGenerateSequence() {
        // 测试序列生成
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        float[][] prompt = {{1, 2, 3}};
        NdArray promptIds = NdArray.of(prompt);
        
        int maxNewTokens = 5;
        NdArray generated = model.generateSequence(promptIds, maxNewTokens);
        
        assertNotNull(generated, "生成序列应存在");
        assertEquals(2, generated.getShape().getDimNum(), 
                    "生成序列应为2维");
        assertEquals(1, generated.getShape().getDimension(0), 
                    "batch维度应为1");
        assertEquals(3 + maxNewTokens, generated.getShape().getDimension(1), 
                    "序列长度应为prompt长度+新生成token数");
    }
    
    @Test
    public void testGenerateSequenceWithDifferentLengths() {
        // 测试不同长度的序列生成
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        float[][] prompt = {{1, 2}};
        NdArray promptIds = NdArray.of(prompt);
        
        // 测试不同的生成长度
        int[] tokenCounts = {1, 3, 5};
        
        for (int maxTokens : tokenCounts) {
            NdArray generated = model.generateSequence(promptIds, maxTokens);
            assertNotNull(generated, "生成长度 " + maxTokens + " 应成功");
            assertEquals(2 + maxTokens, generated.getShape().getDimension(1),
                        "生成长度应正确");
        }
    }
    
    @Test
    public void testModelInfo() {
        // 测试模型信息输出
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        // 打印模型信息（不应抛出异常）
        assertDoesNotThrow(() -> model.printModelInfo(), 
                          "打印模型信息不应抛出异常");
        
        // 验证配置摘要包含关键字
        String summary = model.getConfigSummary();
        assertTrue(summary.contains("DeepSeek-R1"), "摘要应包含模型名称");
        assertTrue(summary.contains("参数"), "摘要应包含参数信息");
        assertTrue(summary.contains("推理"), "摘要应包含推理信息");
    }
    
    @Test
    public void testModelWithDifferentConfigs() {
        // 测试使用不同配置创建模型
        DeepSeekR1Config config1 = DeepSeekR1Config.createTinyConfig();
        DeepSeekR1Config config2 = DeepSeekR1Config.createSmallConfig();
        
        DeepSeekR1Model model1 = new DeepSeekR1Model("model1", config1);
        DeepSeekR1Model model2 = new DeepSeekR1Model("model2", config2);
        
        assertNotNull(model1, "Tiny配置模型应创建成功");
        assertNotNull(model2, "Small配置模型应创建成功");
        
        // 验证不同配置的模型摘要不同
        String summary1 = model1.getConfigSummary();
        String summary2 = model2.getConfigSummary();
        assertNotEquals(summary1, summary2, "不同配置的模型摘要应不同");
    }
    
    @Test
    public void testForwardMethod() {
        // 测试Model基类的forward方法
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        
        float[][] inputData = {{1, 2, 3}};
        Variable input = new Variable(NdArray.of(inputData));
        
        // 调用forward方法（predict内部调用forward）
        Variable output = model.forward(input);
        
        assertNotNull(output, "forward输出应存在");
        assertNotNull(output.getValue(), "forward输出值应存在");
    }
    
    @Test
    public void testReasoningWithDifferentConfidenceThresholds() {
        // 测试不同置信度阈值的推理
        DeepSeekR1Config lowThresholdConfig = DeepSeekR1Config.createTinyConfig();
        lowThresholdConfig.setConfidenceThreshold(0.5); // 低阈值
        
        DeepSeekR1Config highThresholdConfig = DeepSeekR1Config.createTinyConfig();
        highThresholdConfig.setConfidenceThreshold(0.9); // 高阈值
        
        DeepSeekR1Model lowModel = new DeepSeekR1Model("low-threshold", lowThresholdConfig);
        DeepSeekR1Model highModel = new DeepSeekR1Model("high-threshold", highThresholdConfig);
        
        float[][] input = {{1, 2, 3, 4}};
        Variable inputVar = new Variable(NdArray.of(input));
        
        DeepSeekR1Model.ReasoningOutput lowResult = lowModel.performReasoning(inputVar);
        DeepSeekR1Model.ReasoningOutput highResult = highModel.performReasoning(inputVar);
        
        assertNotNull(lowResult, "低阈值推理结果应存在");
        assertNotNull(highResult, "高阈值推理结果应存在");
        
        System.out.println("低阈值(0.5)推理步骤: " + lowResult.numSteps);
        System.out.println("高阈值(0.9)推理步骤: " + highResult.numSteps);
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        DeepSeekR1Model model = new DeepSeekR1Model("test-model", tinyConfig);
        String str = model.toString();
        
        assertNotNull(str, "toString应返回非空字符串");
        assertTrue(str.contains("DeepSeekR1Model"), "toString应包含模型类名");
        
        System.out.println("模型信息: " + str);
    }
}
