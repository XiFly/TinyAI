package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * DeepSeek-R1 RLVR数据集
 * 
 * RLVR (Reinforcement Learning from Verifiable Rewards) 数据集
 * 
 * 与RLHF数据集的区别:
 * - RLHF: 需要人工标注的奖励分数 (0-1连续值)
 * - RLVR: 通过验证器自动获取奖励 (0或1二值)
 * 
 * 数据格式:
 * - 问题 (question)
 * - 标准答案/测试用例 (groundTruth)
 * - 验证器类型 (verifierType)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1RLVRDataset {
    
    private final List<RLVRSample> samples;
    private final int batchSize;
    private final int maxSeqLen;
    private final int vocabSize;
    
    private int currentIndex;
    private List<Integer> shuffledIndices;
    
    /**
     * 构造函数
     * 
     * @param batchSize 批次大小
     * @param maxSeqLen 最大序列长度
     * @param vocabSize 词汇表大小
     */
    public DeepSeekR1RLVRDataset(int batchSize, int maxSeqLen, int vocabSize) {
        this.samples = new ArrayList<>();
        this.batchSize = batchSize;
        this.maxSeqLen = maxSeqLen;
        this.vocabSize = vocabSize;
        this.currentIndex = 0;
        this.shuffledIndices = new ArrayList<>();
    }
    
    /**
     * 添加样本
     * 
     * @param question 问题
     * @param groundTruth 标准答案
     * @param verifierType 验证器类型 ("math", "code", "logic")
     */
    public void addSample(String question, String groundTruth, String verifierType) {
        samples.add(new RLVRSample(question, groundTruth, verifierType));
    }
    
    /**
     * 添加样本（带Token IDs）
     * 
     * @param tokenIds Token ID数组
     * @param groundTruth 标准答案
     * @param verifierType 验证器类型
     */
    public void addSample(float[] tokenIds, String groundTruth, String verifierType) {
        samples.add(new RLVRSample(tokenIds, groundTruth, verifierType));
    }
    
    /**
     * 准备数据集
     * 
     * @param shuffle 是否打乱顺序
     */
    public void prepare(boolean shuffle) {
        currentIndex = 0;
        shuffledIndices.clear();
        
        for (int i = 0; i < samples.size(); i++) {
            shuffledIndices.add(i);
        }
        
        if (shuffle) {
            Collections.shuffle(shuffledIndices);
        }
    }
    
    /**
     * 是否还有下一批次
     */
    public boolean hasNext() {
        return currentIndex < samples.size();
    }
    
    /**
     * 获取下一批次
     */
    public Batch nextBatch() {
        int actualBatchSize = Math.min(batchSize, samples.size() - currentIndex);
        
        float[][] inputIds = new float[actualBatchSize][maxSeqLen];
        String[] questions = new String[actualBatchSize];
        String[] groundTruths = new String[actualBatchSize];
        String[] verifierTypes = new String[actualBatchSize];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int sampleIdx = shuffledIndices.get(currentIndex + i);
            RLVRSample sample = samples.get(sampleIdx);
            
            // 填充input IDs
            float[] tokenIds = sample.getTokenIds();
            System.arraycopy(tokenIds, 0, inputIds[i], 0, 
                Math.min(tokenIds.length, maxSeqLen));
            
            questions[i] = sample.getQuestion();
            groundTruths[i] = sample.getGroundTruth();
            verifierTypes[i] = sample.getVerifierType();
        }
        
        currentIndex += actualBatchSize;
        
        return new Batch(
            NdArray.of(inputIds),
            questions,
            groundTruths,
            verifierTypes
        );
    }
    
    /**
     * 重置数据集
     */
    public void reset() {
        currentIndex = 0;
    }
    
    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 获取批次大小
     */
    public int getBatchSize() {
        return batchSize;
    }
    
    // ==================== 内部类 ====================
    
    /**
     * RLVR样本
     */
    public static class RLVRSample {
        private final String question;
        private final float[] tokenIds;
        private final String groundTruth;
        private final String verifierType;
        
        public RLVRSample(String question, String groundTruth, String verifierType) {
            this.question = question;
            this.tokenIds = simpleTokenize(question);
            this.groundTruth = groundTruth;
            this.verifierType = verifierType;
        }
        
        public RLVRSample(float[] tokenIds, String groundTruth, String verifierType) {
            this.question = "";
            this.tokenIds = tokenIds;
            this.groundTruth = groundTruth;
            this.verifierType = verifierType;
        }
        
        private float[] simpleTokenize(String text) {
            // 简单的tokenization（实际应使用完整的tokenizer）
            if (text == null || text.isEmpty()) {
                return new float[1];
            }
            
            char[] chars = text.toCharArray();
            float[] tokens = new float[Math.min(chars.length, 100)];
            for (int i = 0; i < tokens.length; i++) {
                tokens[i] = (float) (chars[i] % 1000);
            }
            return tokens;
        }
        
        public String getQuestion() {
            return question;
        }
        
        public float[] getTokenIds() {
            return tokenIds;
        }
        
        public String getGroundTruth() {
            return groundTruth;
        }
        
        public String getVerifierType() {
            return verifierType;
        }
    }
    
    /**
     * 批次数据
     */
    public static class Batch {
        private final NdArray inputIds;
        private final String[] questions;
        private final String[] groundTruths;
        private final String[] verifierTypes;
        
        public Batch(NdArray inputIds, String[] questions, 
                    String[] groundTruths, String[] verifierTypes) {
            this.inputIds = inputIds;
            this.questions = questions;
            this.groundTruths = groundTruths;
            this.verifierTypes = verifierTypes;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public String[] getQuestions() {
            return questions;
        }
        
        public String[] getGroundTruths() {
            return groundTruths;
        }
        
        public String[] getVerifierTypes() {
            return verifierTypes;
        }
        
        public int getBatchSize() {
            return questions.length;
        }
    }
}
