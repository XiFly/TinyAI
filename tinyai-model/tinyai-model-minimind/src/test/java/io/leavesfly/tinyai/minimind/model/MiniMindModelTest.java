package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * MiniMind 模型测试
 *
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindModelTest {

    @Test
    public void testModelCreation() {
        // 测试创建 Small 模型
        MiniMindModel model = MiniMindModel.create("test-minimind", "small");
        assertNotNull(model);
        assertNotNull(model.getConfig());
        assertEquals("small", model.getConfig().getModelSize());

        // 打印模型信息
        System.out.println("\n=== Small Model Info ===");
        model.printModelInfo();
    }

    @Test
    public void testModelForward() {
        // 创建 Small 模型
        MiniMindModel model = MiniMindModel.create("test-minimind", "small");
        MiniMindConfig config = model.getConfig();

        // 准备输入：[batch_size=2, seq_len=10]
        int batchSize = 2;
        int seqLen = 10;
        float[] tokenIdsData = new float[batchSize * seqLen];
        for (int i = 0; i < tokenIdsData.length; i++) {
            tokenIdsData[i] = i % config.getVocabSize(); // 随机 token IDs
        }

        NdArray tokenIds = NdArray.of(tokenIdsData, Shape.of(batchSize, seqLen));
        Variable input = new Variable(tokenIds);

        // 前向传播
        Variable output = model.predict(input);

        // 验证输出形状：[batch_size, seq_len, vocab_size]
        assertNotNull(output);
        int[] outputShape = output.getValue().getShape().getShapeDims();
        assertEquals(3, outputShape.length);
        assertEquals(batchSize, outputShape[0]);
        assertEquals(seqLen, outputShape[1]);
        assertEquals(config.getVocabSize(), outputShape[2]);

        System.out.println("\n=== Forward Pass Test ===");
        System.out.println("Input shape: [" + batchSize + ", " + seqLen + "]");
        System.out.println("Output shape: [" + outputShape[0] + ", " + 
                          outputShape[1] + ", " + outputShape[2] + "]");
        System.out.println("Test passed!");
    }

    @Test
    public void testModelGeneration() {
        // 创建 Small 模型
        MiniMindModel model = MiniMindModel.create("test-minimind", "small");

        // 准备提示词 token IDs
        int[] promptTokenIds = {1, 2, 3, 4, 5}; // 简单的提示词

        // 生成文本（贪婪采样）
        int[] generatedTokens = model.generate(
            promptTokenIds,
            10,        // 最大生成 10 个新 token
            0.0f,      // temperature = 0（贪婪）
            0,         // 不使用 top-k
            0.0f       // 不使用 top-p
        );

        // 验证输出
        assertNotNull(generatedTokens);
        assertTrue(generatedTokens.length >= promptTokenIds.length);
        assertTrue(generatedTokens.length <= promptTokenIds.length + 10);

        System.out.println("\n=== Generation Test ===");
        System.out.println("Prompt length: " + promptTokenIds.length);
        System.out.println("Generated length: " + generatedTokens.length);
        System.out.print("Prompt tokens: ");
        for (int i = 0; i < promptTokenIds.length; i++) {
            System.out.print(promptTokenIds[i] + " ");
        }
        System.out.println();
        System.out.print("Generated tokens: ");
        for (int token : generatedTokens) {
            System.out.print(token + " ");
        }
        System.out.println("\nTest passed!");
    }

    @Test
    public void testMediumModelCreation() {
        // 测试创建 Medium 模型
        MiniMindModel model = MiniMindModel.create("test-minimind-medium", "medium");
        assertNotNull(model);
        assertEquals("medium", model.getConfig().getModelSize());

        System.out.println("\n=== Medium Model Info ===");
        model.printModelInfo();
    }

    @Test
    public void testModelConfiguration() {
        // 测试自定义配置
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(1000);
        config.setMaxSeqLen(128);
        config.setHiddenSize(256);
        config.setNumLayers(4);
        config.setNumHeads(8);
        config.setFfnHiddenSize(512);

        MiniMindModel model = new MiniMindModel("custom-minimind", config);
        assertNotNull(model);
        assertEquals(1000, model.getConfig().getVocabSize());
        assertEquals(128, model.getConfig().getMaxSeqLen());
        assertEquals(256, model.getConfig().getHiddenSize());

        System.out.println("\n=== Custom Model Info ===");
        model.printModelInfo();
    }

    @Test
    public void testParameterEstimation() {
        // 测试参数数量估算
        MiniMindModel smallModel = MiniMindModel.create("small", "small");
        long smallParams = smallModel.getConfig().estimateParameters();

        MiniMindModel mediumModel = MiniMindModel.create("medium", "medium");
        long mediumParams = mediumModel.getConfig().estimateParameters();

        // Medium 模型参数应该比 Small 多
        assertTrue(mediumParams > smallParams);

        System.out.println("\n=== ParameterV1 Estimation ===");
        System.out.println("Small model: " + smallParams + " parameters");
        System.out.println("Medium model: " + mediumParams + " parameters");
    }
}
