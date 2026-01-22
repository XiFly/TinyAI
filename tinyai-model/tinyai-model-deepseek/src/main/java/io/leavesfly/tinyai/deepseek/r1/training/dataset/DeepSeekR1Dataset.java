package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DeepSeek-R1数据集类
 * 
 * 支持预训练、后训练和强化学习三种模式的数据加载
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Dataset {
    
    private final List<int[]> sequences;      // 完整序列
    private final List<String> reasoning;     // 推理过程（RLHF用）
    private final List<Float> rewards;        // 奖励分数（RLHF用）
    private final int maxSeqLength;
    private final int batchSize;
    private final boolean shuffle;
    
    private int currentIndex;
    private List<Integer> indices;
    
    /**
     * 构造函数（预训练模式）
     * 
     * @param sequences token序列列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, int maxSeqLength, 
                             int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.reasoning = new ArrayList<>();
        this.rewards = new ArrayList<>();
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 构造函数（RLHF模式）
     * 
     * @param sequences token序列列表
     * @param reasoning 推理过程文本列表
     * @param rewards 奖励分数列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public DeepSeekR1Dataset(List<int[]> sequences, List<String> reasoning,
                             List<Float> rewards, int maxSeqLength,
                             int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.reasoning = reasoning;
        this.rewards = rewards;
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 初始化索引
     */
    private void initIndices() {
        indices = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            indices.add(i);
        }
    }
    
    /**
     * 准备数据集（打乱或重置）
     * 
     * @param shouldShuffle 是否打乱
     */
    public void prepare(boolean shouldShuffle) {
        if (shouldShuffle && shuffle) {
            Collections.shuffle(indices, new Random());
        }
        currentIndex = 0;
    }
    
    /**
     * 是否还有下一批数据
     */
    public boolean hasNext() {
        return currentIndex < sequences.size();
    }
    
    /**
     * 获取下一批数据
     * 
     * @return 批次数据
     */
    public Batch nextBatch() {
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        int actualBatchSize = endIndex - currentIndex;
        
        // 准备输入和目标
        float[][] inputData = new float[actualBatchSize][maxSeqLength];
        float[][] targetData = new float[actualBatchSize][maxSeqLength];
        String[] reasoningTexts = new String[actualBatchSize];
        float[] rewardScores = new float[actualBatchSize];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = indices.get(currentIndex + i);
            int[] sequence = sequences.get(dataIndex);
            
            // 填充或截断序列
            int seqLen = Math.min(sequence.length, maxSeqLength);
            
            // 输入：序列的前n-1个token
            for (int j = 0; j < seqLen - 1; j++) {
                inputData[i][j] = sequence[j];
            }
            
            // 目标：序列的后n-1个token（用于语言建模）
            for (int j = 1; j < seqLen; j++) {
                targetData[i][j - 1] = sequence[j];
            }
            
            // RLHF数据
            if (!reasoning.isEmpty() && dataIndex < reasoning.size()) {
                reasoningTexts[i] = reasoning.get(dataIndex);
            }
            if (!rewards.isEmpty() && dataIndex < rewards.size()) {
                rewardScores[i] = rewards.get(dataIndex);
            }
        }
        
        currentIndex = endIndex;
        
        NdArray inputIds = NdArray.of(inputData);
        NdArray targetIds = NdArray.of(targetData);
        
        return new Batch(inputIds, targetIds, reasoningTexts, rewardScores);
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
        return sequences.size();
    }
    
    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return (sequences.size() + batchSize - 1) / batchSize;
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final String[] reasoning;
        private final float[] rewards;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    String[] reasoning, float[] rewards) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.reasoning = reasoning;
            this.rewards = rewards;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public NdArray getTargetIds() {
            return targetIds;
        }
        
        public String[] getReasoning() {
            return reasoning;
        }
        
        public float[] getRewards() {
            return rewards;
        }
        
        public int getBatchSize() {
            return inputIds.getShape().getDimension(0);
        }
    }
    
    /**
     * 创建示例数据集（用于测试）
     */
    public static DeepSeekR1Dataset createDummyDataset(int numSamples, int seqLength,
                                                        int vocabSize, int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] seq = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                seq[j] = random.nextInt(vocabSize);
            }
            sequences.add(seq);
        }
        
        return new DeepSeekR1Dataset(sequences, seqLength, batchSize, true);
    }
    
    /**
     * 创建RLHF示例数据集
     */
    public static DeepSeekR1Dataset createDummyRLHFDataset(int numSamples, int seqLength,
                                                            int vocabSize, int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        List<String> reasoning = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int[] seq = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                seq[j] = random.nextInt(vocabSize);
            }
            sequences.add(seq);
            
            reasoning.add("推理步骤" + i);
            rewards.add(random.nextFloat());
        }
        
        return new DeepSeekR1Dataset(sequences, reasoning, rewards, 
                                     seqLength, batchSize, true);
    }
}
