package io.leavesfly.tinyai.deepseek.rx;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeekR1Config单元测试
 * 
 * 测试范围：
 * 1. 默认配置创建
 * 2. 预设配置工厂方法（Tiny、Small、Standard）
 * 3. 配置参数验证
 * 4. 参数量估算
 * 5. 配置setter/getter
 * 
 * @author leavesfly
 */
public class DeepSeekR1ConfigTest {
    
    @Test
    public void testDefaultConfig() {
        // 测试默认配置创建
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 验证基础配置默认值
        assertEquals(50257, config.getVocabSize(), "默认词汇表大小应为50257");
        assertEquals(2048, config.getNPositions(), "默认序列长度应为2048");
        assertEquals(768, config.getNEmbd(), "默认嵌入维度应为768");
        assertEquals(12, config.getNLayer(), "默认层数应为12");
        assertEquals(12, config.getNHead(), "默认注意力头数应为12");
        assertEquals(3072, config.getNInner(), "默认前馈网络维度应为3072");
        
        // 验证推理配置默认值
        assertEquals(7, config.getMaxReasoningSteps(), "默认推理步骤应为7");
        assertEquals(0.7, config.getConfidenceThreshold(), 0.001, 
                    "默认置信度阈值应为0.7");
        
        // 验证反思配置
        assertEquals(5, config.getQualityScoreDim(), "默认质量评分维度应为5");
        assertEquals(3, config.getMaxSuggestions(), "默认最大建议数应为3");
    }
    
    @Test
    public void testTinyConfig() {
        // 测试微型配置
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        
        assertEquals(10000, config.getVocabSize(), "Tiny配置词汇表大小应为10000");
        assertEquals(256, config.getNEmbd(), "Tiny配置嵌入维度应为256");
        assertEquals(6, config.getNLayer(), "Tiny配置层数应为6");
        assertEquals(8, config.getNHead(), "Tiny配置注意力头数应为8");
        assertEquals(512, config.getNPositions(), "Tiny配置序列长度应为512");
        assertEquals(5, config.getMaxReasoningSteps(), "Tiny配置推理步骤应为5");
    }
    
    @Test
    public void testSmallConfig() {
        // 测试小型配置
        DeepSeekR1Config config = DeepSeekR1Config.createSmallConfig();
        
        assertEquals(30000, config.getVocabSize(), "Small配置词汇表大小应为30000");
        assertEquals(512, config.getNEmbd(), "Small配置嵌入维度应为512");
        assertEquals(8, config.getNLayer(), "Small配置层数应为8");
        assertEquals(8, config.getNHead(), "Small配置注意力头数应为8");
        assertEquals(1024, config.getNPositions(), "Small配置序列长度应为1024");
        assertEquals(6, config.getMaxReasoningSteps(), "Small配置推理步骤应为6");
    }
    
    @Test
    public void testStandardConfig() {
        // 测试标准配置
        DeepSeekR1Config config = DeepSeekR1Config.createStandardConfig();
        
        assertEquals(50257, config.getVocabSize(), "Standard配置词汇表大小应为50257");
        assertEquals(768, config.getNEmbd(), "Standard配置嵌入维度应为768");
        assertEquals(12, config.getNLayer(), "Standard配置层数应为12");
        assertEquals(12, config.getNHead(), "Standard配置注意力头数应为12");
        assertEquals(2048, config.getNPositions(), "Standard配置序列长度应为2048");
        assertEquals(7, config.getMaxReasoningSteps(), "Standard配置推理步骤应为7");
    }
    
    @Test
    public void testConfigValidation() {
        // 测试配置验证
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 默认配置应该通过验证
        assertDoesNotThrow(() -> config.validate(), "默认配置应该有效");
        
        // 测试无效的词汇表大小
        config.setVocabSize(-1);
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "负数词汇表大小应该抛出异常");
        
        // 恢复有效值
        config.setVocabSize(50257);
        
        // 测试无效的嵌入维度
        config.setNEmbd(0);
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "零嵌入维度应该抛出异常");
        
        // 恢复有效值
        config.setNEmbd(768);
        
        // 测试无效的注意力头数（嵌入维度不能被头数整除）
        config.setNHead(7); // 768不能被7整除
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "嵌入维度不能被注意力头数整除时应该抛出异常");
        
        // 恢复有效值
        config.setNHead(12);
        
        // 测试无效的推理步骤数
        config.setMaxReasoningSteps(-1);
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "负数推理步骤应该抛出异常");
    }
    
    @Test
    public void testParameterCountEstimation() {
        // 测试参数量估算
        DeepSeekR1Config tinyConfig = DeepSeekR1Config.createTinyConfig();
        DeepSeekR1Config standardConfig = DeepSeekR1Config.createStandardConfig();
        
        long tinyParams = tinyConfig.estimateParameterCount();
        long standardParams = standardConfig.estimateParameterCount();
        
        // Tiny配置参数量应该更小
        assertTrue(tinyParams < standardParams, 
                  "Tiny配置的参数量应该小于Standard配置");
        
        // 参数量应该大于0
        assertTrue(tinyParams > 0, "Tiny配置参数量应大于0");
        assertTrue(standardParams > 0, "Standard配置参数量应大于0");
        
        // 打印参数量信息（用于调试）
        System.out.println("Tiny配置参数量: " + formatParamCount(tinyParams));
        System.out.println("Standard配置参数量: " + formatParamCount(standardParams));
    }
    
    @Test
    public void testSettersAndGetters() {
        // 测试所有setter和getter方法
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 基础配置
        config.setVocabSize(100000);
        assertEquals(100000, config.getVocabSize());
        
        config.setNEmbd(1024);
        assertEquals(1024, config.getNEmbd());
        
        config.setNLayer(16);
        assertEquals(16, config.getNLayer());
        
        config.setNHead(16);
        assertEquals(16, config.getNHead());
        
        // 推理配置
        config.setMaxReasoningSteps(10);
        assertEquals(10, config.getMaxReasoningSteps());
        
        config.setConfidenceThreshold(0.8);
        assertEquals(0.8, config.getConfidenceThreshold(), 0.001);
        
        config.setReasoningHiddenDim(2048);
        assertEquals(2048, config.getReasoningHiddenDim());
        
        // 反思配置
        config.setReflectionHiddenDim(2048);
        assertEquals(2048, config.getReflectionHiddenDim());
        
        config.setQualityScoreDim(7);
        assertEquals(7, config.getQualityScoreDim());
        
        config.setMaxSuggestions(5);
        assertEquals(5, config.getMaxSuggestions());
        
        // Dropout配置
        config.setResidPdrop(0.2);
        assertEquals(0.2, config.getResidPdrop(), 0.0001);
        
        config.setEmbdPdrop(0.15);
        assertEquals(0.15, config.getEmbdPdrop(), 0.0001);
        
        config.setAttnPdrop(0.1);
        assertEquals(0.1, config.getAttnPdrop(), 0.0001);
    }
    
    @Test
    public void testReasoningConfiguration() {
        // 测试推理特定配置
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 设置推理配置
        config.setMaxReasoningSteps(7);
        config.setReasoningHiddenDim(1536);
        config.setConfidenceThreshold(0.7);
        
        assertEquals(7, config.getMaxReasoningSteps(), "推理步骤应为7");
        assertEquals(1536, config.getReasoningHiddenDim(), 
                    "推理隐藏层维度应为1536");
        assertEquals(0.7, config.getConfidenceThreshold(), 0.001, 
                    "置信度阈值应为0.7");
    }
    
    @Test
    public void testReflectionConfiguration() {
        // 测试反思特定配置
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 设置反思配置
        config.setReflectionHiddenDim(1536);
        config.setQualityScoreDim(5);
        config.setMaxSuggestions(3);
        
        assertEquals(1536, config.getReflectionHiddenDim(), 
                    "反思隐藏层维度应为1536");
        assertEquals(5, config.getQualityScoreDim(), 
                    "质量评分维度应为5");
        assertEquals(3, config.getMaxSuggestions(), 
                    "最大建议数应为3");
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        DeepSeekR1Config config = DeepSeekR1Config.createTinyConfig();
        String str = config.toString();
        
        assertNotNull(str, "toString应返回非空字符串");
        assertTrue(str.contains("DeepSeekR1Config"), "toString应包含类名");
        assertTrue(str.contains("vocabSize"), "toString应包含vocabSize");
        assertTrue(str.contains("nEmbd"), "toString应包含nEmbd");
        assertTrue(str.contains("maxReasoningSteps"), "toString应包含maxReasoningSteps");
        
        System.out.println("配置信息:\n" + str);
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
}
