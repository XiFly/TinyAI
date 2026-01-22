package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.MultiHeadAttention;

/**
 * DeepSeek-R1 Transformer块（Pre-LayerNorm架构）
 * 
 * 采用Pre-LN架构提升训练稳定性：LayerNorm -> SubLayer -> Dropout -> Add
 * 包含两个子层：
 * 1. 多头自注意力层（带因果掩码）
 * 2. 前馈神经网络层
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1TransformerBlock extends Module {
    
    private final DeepSeekR1Config config;
    
    // 注意力子层
    private final MultiHeadAttention attention;
    private final LayerNorm layerNorm1;
    private final Dropout attnDropout;
    
    // 前馈子层
    private final Linear ffnLinear1;
    private final GELU activation;
    private final Linear ffnLinear2;
    private final LayerNorm layerNorm2;
    private final Dropout mlpDropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1TransformerBlock(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        
        int dModel = config.getNEmbd();
        int numHeads = config.getNHead();
        int dFF = config.getNInner();
        float dropout = (float) config.getResidPdrop();
        float attnDropoutRate = (float) config.getAttnPdrop();
        
        // 初始化注意力子层组件
        this.attention = new MultiHeadAttention("attn", dModel, numHeads, attnDropoutRate);
        this.layerNorm1 = new LayerNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        this.attnDropout = new Dropout("attn_dropout", dropout);
        
        // 初始化前馈子层组件
        this.ffnLinear1 = new Linear("ffn_fc1", dModel, dFF, true);
        this.activation = new GELU("gelu");
        this.ffnLinear2 = new Linear("ffn_fc2", dFF, dModel, true);
        this.layerNorm2 = new LayerNorm("ln2", dModel, (float) config.getLayerNormEpsilon());
        this.mlpDropout = new Dropout("mlp_dropout", dropout);
        
        // 注册所有子模块
        registerModule("attn", attention);
        registerModule("ln1", layerNorm1);
        registerModule("attn_dropout", attnDropout);
        registerModule("ffn_fc1", ffnLinear1);
        registerModule("gelu", activation);
        registerModule("ffn_fc2", ffnLinear2);
        registerModule("ln2", layerNorm2);
        registerModule("mlp_dropout", mlpDropout);
    }
    
    /**
     * 前向传播
     * 
     * Pre-LN架构流程：
     * 1. 注意力分支: x -> LN -> Attn -> Dropout -> Add(x)
     * 2. 前馈分支: x -> LN -> FFN -> Dropout -> Add(x)
     * 
     * @param inputs 输入变量，inputs[0]为输入张量 [batch_size, seq_len, d_model]
     * @return 输出张量 [batch_size, seq_len, d_model]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable x = inputs[0];
        int seqLen = x.getValue().getShape().getDimension(1);
        
        // 生成因果掩码（下三角矩阵）
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(seqLen);
        
        // ===== 注意力子层 (Pre-LN) =====
        // LN -> MultiHeadAttention -> Dropout -> Add
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(normalized1, normalized1, normalized1, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);
        
        // ===== 前馈子层 (Pre-LN) =====
        // LN -> FFN -> Dropout -> Add
        Variable normalized2 = layerNorm2.forward(residual1);
        Variable mlpOutput = ffnLinear1.forward(normalized2);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);
        Variable output = residual1.add(mlpOutput);
        
        return output;
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
}
