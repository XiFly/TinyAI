package io.leavesfly.tinyai.deepseek.r1.training.verifier;

/**
 * 验证器接口
 * 
 * RLVR (Reinforcement Learning from Verifiable Rewards) 的核心组件
 * 
 * 与RLHF的区别：
 * - RLHF: 基于人类主观反馈，奖励是连续值
 * - RLVR: 基于客观规则验证，奖励是二值(0/1)
 * 
 * 适用场景：
 * - 数学计算（可验证答案正确性）
 * - 代码执行（可通过测试用例验证）
 * - 逻辑推理（可验证推理链条）
 * 
 * @author leavesfly
 * @version 1.0
 */
public interface Verifier {
    
    /**
     * 验证模型输出是否正确
     * 
     * @param modelOutput 模型生成的输出（包含推理过程和答案）
     * @param groundTruth 标准答案或测试用例
     * @return 验证结果对象
     */
    VerificationResult verify(String modelOutput, String groundTruth);
    
    /**
     * 获取验证器类型
     * 
     * @return 验证器类型名称（如 "math", "code", "logic"）
     */
    String getVerifierType();
    
    /**
     * 从模型输出中提取答案
     * 
     * 不同验证器可能有不同的答案提取策略
     * 例如：
     * - 数学验证器: 提取最后的数值
     * - 代码验证器: 提取代码块
     * - 逻辑验证器: 提取结论
     * 
     * @param modelOutput 模型输出
     * @return 提取的答案
     */
    String extractAnswer(String modelOutput);
}
