package io.leavesfly.tinyai.nnet.v2.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * V2版本的TransformerEncoderLayer
 * <p>
 * Transformer编码器层，包含：
 * 1. 多头自注意力子层
 * 2. 前馈神经网络子层
 * 3. 两个层归一化层
 * 4. 残差连接
 * <p>
 * 结构：
 * x -> LayerNorm -> MultiHeadAttention -> Add(x) ->
 * -> LayerNorm -> FFN -> Add -> output
 * <p>
 * 或者（Post-LN）：
 * x -> MultiHeadAttention -> Add(x) -> LayerNorm ->
 * -> FFN -> Add -> LayerNorm -> output
 *
 * @author leavesfly
 * @version 2.0
 */
public class TransformerEncoderLayer extends Module {

    private final int dModel;
    private final int numHeads;
    private final int dFF;
    private final float dropout;
    private final boolean preLayerNorm;

    // 子模块
    private MultiHeadAttention selfAttention;
    private LayerNorm norm1;
    private Linear ffn1;
    private ReLU activation;
    private Linear ffn2;
    private LayerNorm norm2;

    /**
     * 构造函数
     *
     * @param name         层名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param dFF          前馈网络隐藏层维度
     * @param dropout      dropout比率
     * @param preLayerNorm 是否使用Pre-LayerNorm（true: Pre-LN, false: Post-LN）
     */
    public TransformerEncoderLayer(String name, int dModel, int numHeads, int dFF,
                                   float dropout, boolean preLayerNorm) {
        super(name);
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropout = dropout;
        this.preLayerNorm = preLayerNorm;

        // 初始化子模块
        selfAttention = new MultiHeadAttention("self_attn", dModel, numHeads, dropout);
        norm1 = new LayerNorm("norm1", dModel);

        // 前馈网络: d_model -> d_ff -> d_model
        ffn1 = new Linear("ffn1", dModel, dFF, true);
        activation = new ReLU();
        ffn2 = new Linear("ffn2", dFF, dModel, true);
        norm2 = new LayerNorm("norm2", dModel);

        // 注册子模块
        registerModule("self_attn", selfAttention);
        registerModule("norm1", norm1);
        registerModule("ffn1", ffn1);
        registerModule("activation", activation);
        registerModule("ffn2", ffn2);
        registerModule("norm2", norm2);

        init();
    }

    /**
     * 构造函数（使用默认参数）
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     */
    public TransformerEncoderLayer(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, dModel * 4, 0.1f, true);
    }

    /**
     * 构造函数
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param dFF      前馈网络隐藏层维度
     */
    public TransformerEncoderLayer(String name, int dModel, int numHeads, int dFF) {
        this(name, dModel, numHeads, dFF, 0.1f, true);
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        if (preLayerNorm) {
            // Pre-LayerNorm
            x = forwardPreNorm(x);
        } else {
            // Post-LayerNorm
            x = forwardPostNorm(x);
        }

        return x;
    }

    /**
     * Pre-LayerNorm前向传播
     * <p>
     * x -> LayerNorm -> Attention -> Add(x) -> LayerNorm -> FFN -> Add(x)
     *
     * @param x 输入
     * @return 输出
     */
    private Variable forwardPreNorm(Variable x) {
        // 自注意力子层（Pre-LN）
        Variable norm_x = norm1.forward(x);
        Variable attn_out = selfAttention.forward(norm_x, norm_x, norm_x);
        Variable residual1 = x.add(attn_out);

        // 前馈网络子层（Pre-LN）
        Variable norm_residual1 = norm2.forward(residual1);
        Variable ffn_out = forwardFFN(norm_residual1);
        Variable output = residual1.add(ffn_out);

        return output;
    }

    /**
     * Post-LayerNorm前向传播
     * <p>
     * x -> Attention -> Add(x) -> LayerNorm -> FFN -> Add(x) -> LayerNorm
     *
     * @param x 输入
     * @return 输出
     */
    private Variable forwardPostNorm(Variable x) {
        // 自注意力子层（Post-LN）
        Variable attn_out = selfAttention.forward(x, x, x);
        Variable residual1 = x.add(attn_out);
        Variable norm1_out = norm1.forward(residual1);

        // 前馈网络子层（Post-LN）
        Variable ffn_out = forwardFFN(norm1_out);
        Variable residual2 = norm1_out.add(ffn_out);
        Variable output = norm2.forward(residual2);

        return output;
    }

    /**
     * 前馈网络前向传播
     * <p>
     * FFN(x) = ReLU(x W1 + b1) W2 + b2
     *
     * @param x 输入
     * @return 输出
     */
    private Variable forwardFFN(Variable x) {
        Variable h = ffn1.forward(x);
        h = activation.forward(h);
        Variable output = ffn2.forward(h);
        return output;
    }

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDFF() {
        return dFF;
    }

    public float getDropout() {
        return dropout;
    }

    public boolean isPreLayerNorm() {
        return preLayerNorm;
    }

    @Override
    public String toString() {
        return "TransformerEncoderBlock{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", numHeads=" + numHeads +
                ", dFF=" + dFF +
                ", dropout=" + dropout +
                ", preLayerNorm=" + preLayerNorm +
                '}';
    }
}
