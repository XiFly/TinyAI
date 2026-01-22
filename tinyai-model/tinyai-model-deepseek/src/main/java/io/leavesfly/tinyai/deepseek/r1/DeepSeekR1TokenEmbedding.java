package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * DeepSeek-R1 Token嵌入层
 * 
 * 负责将输入的token ID序列转换为稠密的向量表示，包括：
 * 1. Token嵌入 - 词汇表中每个token的向量表示
 * 2. 位置嵌入 - 序列中每个位置的向量表示
 * 3. Dropout - 嵌入层的正则化
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1TokenEmbedding extends Module {
    
    private final int vocabSize;
    private final int embeddingDim;
    private final int maxPositions;
    private final float dropoutProb;
    
    private Parameter tokenEmbedding;
    private Parameter positionEmbedding;
    private Dropout dropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1TokenEmbedding(String name, DeepSeekR1Config config) {
        super(name);
        this.vocabSize = config.getVocabSize();
        this.embeddingDim = config.getNEmbd();
        this.maxPositions = config.getNPositions();
        this.dropoutProb = (float) config.getEmbdPdrop();
        initializeParameters(config);
    }
    
    /**
     * 初始化嵌入层参数
     * 
     * @param config R1配置对象
     */
    private void initializeParameters(DeepSeekR1Config config) {
        // 初始化token嵌入矩阵 [vocabSize, embeddingDim]
        NdArray tokenEmbedData = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        tokenEmbedding = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbedding);
        
        // 初始化位置嵌入矩阵 [maxPositions, embeddingDim]
        NdArray positionEmbedData = NdArray.likeRandomN(Shape.of(maxPositions, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        positionEmbedding = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbedding);
        
        // 初始化dropout层
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs 输入变量，inputs[0]为token ID序列 [batch_size, seq_len]
     * @return 嵌入向量 [batch_size, seq_len, embeddingDim]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];
        NdArray tokenData = tokenIds.getValue();
        
        // 验证输入维度
        if (tokenData.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", tokenData.getShape())
            );
        }
        
        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);
        
        // 验证序列长度
        if (sequenceLength > maxPositions) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", sequenceLength, maxPositions)
            );
        }
        
        // ✅ 使用Variable层面的算子
        Variable tokenEmbedParam = new Variable(tokenEmbedding.data());
        Variable tokenEmbeds = getTokenEmbeddingsV2(tokenIds, tokenEmbedParam, batchSize, sequenceLength);
        
        Variable posEmbedParam = new Variable(positionEmbedding.data());
        Variable positionEmbeds = getPositionEmbeddingsV2(posEmbedParam, batchSize, sequenceLength);
        
        // 合并嵌入并应用dropout
        Variable combined = tokenEmbeds.add(positionEmbeds);
        return dropout.forward(combined);
    }
    
    /**
     * 获取token嵌入向量 (使用Variable算子)
     * 
     * @param tokenIds token ID变量
     * @param tokenEmbedParam token嵌入参数
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return token嵌入变量 [batch_size, seq_len, embeddingDim]
     */
    private Variable getTokenEmbeddingsV2(Variable tokenIds, Variable tokenEmbedParam,
                                          int batchSize, int sequenceLength) {
        // ✅ 使用indexSelect算子
        Variable flatIds = tokenIds.reshape(Shape.of(batchSize * sequenceLength));
        Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatIds);
        return flatEmbeds.reshape(Shape.of(batchSize, sequenceLength, embeddingDim));
    }
    
    /**
     * 获取位置嵌入向量 (使用Variable算子)
     * 
     * @param posEmbedParam 位置嵌入参数
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return 位置嵌入变量 [1, seq_len, embeddingDim] - 依赖广播机制自动扩展
     */
    private Variable getPositionEmbeddingsV2(Variable posEmbedParam, int batchSize, int sequenceLength) {
        // ✅ 使用indexSelect算子，不使用repeat以节省内存
        // 返回 [1, seq_len, embeddingDim] 形状，依赖add的广播机制自动扩展
        float[] posIndices = new float[sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            posIndices[i] = i;
        }
        Variable posIds = new Variable(NdArray.of(posIndices));
        posIds.setRequireGrad(false);
        Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);
        return posEmbeds.reshape(Shape.of(1, sequenceLength, embeddingDim));
    }
    
    /**
     * 获取token嵌入参数
     */
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    /**
     * 获取位置嵌入参数
     */
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
}
