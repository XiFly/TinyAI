package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.dataset.DPODataset;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DPO端到端测试
 * 
 * 测试DPO训练的完整流程
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOE2ETest {
    
    private MiniMindModel model;
    private MiniMindTokenizer tokenizer;
    private DPODataset dataset;
    private DPOConfig dpoConfig;
    
    @BeforeEach
    public void setUp() {
        // 创建小型测试模型
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(100);
        config.setMaxSeqLen(32);
        config.setHiddenSize(64);
        config.setNumLayers(2);
        config.setNumHeads(2);
        config.setFfnHiddenSize(128);
        config.setDropout(0.0f);
        
        model = new MiniMindModel("test_model", config);
        tokenizer = MiniMindTokenizer.createCharLevelTokenizer(config.getVocabSize(), config.getMaxSeqLen());
        
        // 创建DPO配置
        dpoConfig = DPOConfig.createDefault();
        dpoConfig.setBeta(0.1f);
    }
    
    @Test
    public void testDPODatasetCreation() {
        dataset = new DPODataset(tokenizer, 32, 2);
        
        // 添加偏好对
        dataset.addSample(
            "What is AI?",
            " Artificial Intelligence is a field of computer science.",
            " I don't know."
        );
        
        dataset.addSample(
            "How to learn programming?",
            " Start with basics and practice regularly.",
            " Just Google it."
        );
        
        assertEquals(2, dataset.getSampleCount());
        
        // 准备批次
        dataset.prepare(false);
        assertTrue(dataset.getBatchCount() > 0);
    }
    
    @Test
    public void testDPOConfigValidation() {
        DPOConfig config = new DPOConfig();
        
        // 默认配置应该有效
        assertDoesNotThrow(() -> config.validate());
        
        // 无效的beta
        config.setBeta(-0.1f);
        assertThrows(IllegalArgumentException.class, () -> config.validate());
        
        // 恢复有效值
        config.setBeta(0.1f);
        assertDoesNotThrow(() -> config.validate());
        
        // 无效的标签平滑
        config.setLabelSmoothing(1.5f);
        assertThrows(IllegalArgumentException.class, () -> config.validate());
    }
    
    @Test
    public void testDPOConfigPresets() {
        // 默认配置
        DPOConfig defaultConfig = DPOConfig.createDefault();
        assertEquals(0.1f, defaultConfig.getBeta(), 0.001f);
        assertTrue(defaultConfig.isResponseOnlyLoss());
        
        // 保守配置
        DPOConfig conservative = DPOConfig.createConservative();
        assertEquals(0.5f, conservative.getBeta(), 0.001f);
        assertEquals(0.1f, conservative.getLabelSmoothing(), 0.001f);
        assertTrue(conservative.isUseLengthNormalization());
        
        // 激进配置
        DPOConfig aggressive = DPOConfig.createAggressive();
        assertEquals(0.05f, aggressive.getBeta(), 0.001f);
        assertEquals(0.0f, aggressive.getLabelSmoothing(), 0.001f);
        assertFalse(aggressive.isUseLengthNormalization());
    }
    
    @Test
    public void testDPOTrainerCreation() {
        dataset = new DPODataset(tokenizer, 32, 2);
        dataset.addSample("Q:", " A1", " A2");
        dataset.addSample("Q:", " B1", " B2");
        dataset.prepare(false);
        
        // 创建训练器
        DPOTrainer trainer = new DPOTrainer(model, dataset, dpoConfig);
        assertNotNull(trainer);
        
        // 配置训练参数
        trainer.configure(1, 1e-5f, 1.0f);
        trainer.setCheckpoint("./test_checkpoints", 100);
    }
    
    @Test
    public void testDPOTrainingSingleEpoch() {
        dataset = new DPODataset(tokenizer, 32, 2);
        
        // 添加多个偏好对
        for (int i = 0; i < 10; i++) {
            dataset.addSample(
                "Question " + i + ":",
                " Good answer " + i,
                " Bad answer " + i
            );
        }
        
        dataset.prepare(false);
        
        // 创建并配置训练器
        DPOTrainer trainer = new DPOTrainer(model, dataset, dpoConfig);
        trainer.configure(1, 1e-5f, 1.0f);  // 单个epoch
        
        // 训练应该成功完成
        assertDoesNotThrow(() -> trainer.train());
        
        // 检查训练历史
        assertFalse(trainer.getLossHistory().isEmpty());
        assertFalse(trainer.getAccuracyHistory().isEmpty());
        
        System.out.println("训练完成,损失历史: " + trainer.getLossHistory());
        System.out.println("准确率历史: " + trainer.getAccuracyHistory());
    }
    
    @Test
    public void testDPOBatchProcessing() {
        dataset = new DPODataset(tokenizer, 32, 2);
        
        // 添加样本
        dataset.addSample("P1:", " C1", " R1");
        dataset.addSample("P2:", " C2", " R2");
        dataset.addSample("P3:", " C3", " R3");
        dataset.addSample("P4:", " C4", " R4");
        
        dataset.prepare(false);
        
        int batchCount = 0;
        while (dataset.hasNext()) {
            DPODataset.Batch batch = dataset.nextBatch();
            assertNotNull(batch);
            assertNotNull(batch.getChosenInput());
            assertNotNull(batch.getRejectedInput());
            assertNotNull(batch.getPromptMask());
            batchCount++;
        }
        
        assertEquals(dataset.getBatchCount(), batchCount);
    }
    
    @Test
    public void testDPOLossComputation() {
        DPOLoss loss = new DPOLoss(0.1f, 0.0f);
        assertNotNull(loss);
        
        // 损失计算逻辑在训练器中测试
        // 这里只验证创建成功
    }
    
    @Test
    public void testDifferentBetaValues() {
        dataset = new DPODataset(tokenizer, 32, 2);
        dataset.addSample("Q:", " A1", " A2");
        dataset.prepare(false);
        
        // 测试不同beta值
        float[] betaValues = {0.05f, 0.1f, 0.3f, 0.5f};
        
        for (float beta : betaValues) {
            DPOConfig config = new DPOConfig();
            config.setBeta(beta);
            
            DPOTrainer trainer = new DPOTrainer(model, dataset, config);
            assertNotNull(trainer);
            
            System.out.println("Beta = " + beta + " 的训练器创建成功");
        }
    }
}
