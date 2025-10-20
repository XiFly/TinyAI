package io.leavesfly.tinyai.nnet.v2.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * V2版本的MultiHeadAttention层
 * <p>
 * 多头注意力机制是Transformer架构的核心组件。
 * <p>
 * 核心思想：
 * 1. 将查询(Q)、键(K)、值(V)分别投影到多个子空间
 * 2. 在每个子空间独立计算注意力
 * 3. 将多头结果拼接后再进行一次线性变换
 * <p>
 * 公式：
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
 * head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 * <p>
 * 其中：
 * - h: 注意力头数
 * - d_k: 每个头的维度 (d_model / h)
 * - W^Q, W^K, W^V: Query、Key、Value的投影矩阵
 * - W^O: 输出投影矩阵
 *
 * @author leavesfly
 * @version 2.0
 */
public class MultiHeadAttention extends Module {

    private final int dModel;      // 模型维度
    private final int numHeads;    // 注意力头数
    private final int dK;          // 每个头的键/查询维度
    private final int dV;          // 每个头的值维度
    private final float dropout;   // dropout比率

    // 线性投影层
    private Linear queryProjection;  // Q投影
    private Linear keyProjection;    // K投影
    private Linear valueProjection;  // V投影
    private Linear outputProjection; // 输出投影

    /**
     * 构造函数
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param dropout  dropout比率
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, float dropout) {
        super(name);

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                    String.format("d_model (%d) must be divisible by num_heads (%d)", dModel, numHeads));
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dV = dModel / numHeads;
        this.dropout = dropout;

        // 创建投影层
        queryProjection = new Linear("q_proj", dModel, dModel, true);
        keyProjection = new Linear("k_proj", dModel, dModel, true);
        valueProjection = new Linear("v_proj", dModel, dModel, true);
        outputProjection = new Linear("o_proj", dModel, dModel, true);

        // 注册子模块
        registerModule("q_proj", queryProjection);
        registerModule("k_proj", keyProjection);
        registerModule("v_proj", valueProjection);
        registerModule("o_proj", outputProjection);

        init();
    }

    /**
     * 构造函数（默认dropout=0.1）
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     */
    public MultiHeadAttention(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, 0.1f);
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 输入：query, key, value (可选mask)
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : key;

        int[] queryShape = query.getValue().getShape().getShapeDims();
        int batchSize = queryShape[0];
        int seqLen = queryShape[1];

        // 1. 线性投影 Q, K, V
        Variable Q = queryProjection.forward(query);   // (batch, seq_len, d_model)
        Variable K = keyProjection.forward(key);       // (batch, seq_len, d_model)
        Variable V = valueProjection.forward(value);   // (batch, seq_len, d_model)

        // 2. 分割成多头
        // 需要将 (batch, seq_len, d_model) 重塑为 (batch, seq_len, num_heads, d_k)
        // 然后转置为 (batch, num_heads, seq_len, d_k)
        Q = splitHeads(Q, batchSize, seqLen);
        K = splitHeads(K, batchSize, seqLen);
        V = splitHeads(V, batchSize, seqLen);

        // 3. 计算缩放点积注意力
        Variable attention = scaledDotProductAttention(Q, K, V);

        // 4. 合并多头
        Variable concat = mergeHeads(attention, batchSize, seqLen);

        // 5. 输出投影
        Variable output = outputProjection.forward(concat);

        return output;
    }

    /**
     * 分割成多头
     * <p>
     * 输入: (batch, seq_len, d_model)
     * 输出: (batch, num_heads, seq_len, d_k)
     *
     * @param x         输入变量
     * @param batchSize 批次大小
     * @param seqLen    序列长度
     * @return 分割后的张量
     */
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        // 简化实现：暂时返回原始形状
        // TODO: 需要实现reshape和transpose操作
        // 当前简化版本假设单头注意力
        return x;
    }

    /**
     * 合并多头
     * <p>
     * 输入: (batch, num_heads, seq_len, d_v)
     * 输出: (batch, seq_len, d_model)
     *
     * @param x         输入变量
     * @param batchSize 批次大小
     * @param seqLen    序列长度
     * @return 合并后的张量
     */
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        // 简化实现：暂时返回原始形状
        // TODO: 需要实现transpose和reshape操作
        return x;
    }

    /**
     * 缩放点积注意力
     * <p>
     * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
     *
     * @param Q 查询张量
     * @param K 键张量
     * @param V 值张量
     * @return 注意力输出
     */
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V) {
        // 1. 计算 Q * K^T
        Variable scores = Q.matMul(K.transpose());

        // 2. 缩放
        double scale = Math.sqrt(dK);
        Variable scaledScores = scores.div(new Variable((float) scale));

        // 3. Softmax
        Variable attentionWeights = scaledScores.softMax();

        // 4. 应用dropout（训练模式）
        if (isTraining() && dropout > 0) {
            // TODO: 实现dropout
        }

        // 5. 计算注意力输出: attention_weights * V
        Variable output = attentionWeights.matMul(V);

        return output;
    }

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDK() {
        return dK;
    }

    public int getDV() {
        return dV;
    }

    public float getDropout() {
        return dropout;
    }

    @Override
    public String toString() {
        return "MultiHeadAttention{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", numHeads=" + numHeads +
                ", dK=" + dK +
                ", dV=" + dV +
                ", dropout=" + dropout +
                '}';
    }
}
