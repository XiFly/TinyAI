package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.ParameterV1;
import io.leavesfly.tinyai.nnet.RnnLayer;

import java.util.List;
import java.util.Objects;

/**
 * 简单递归网络层实现 (Simple RNN Layer)
 * <p>
 * 这是一个使用 tanh 作为激活函数的标准循环神经网络层实现。
 * 该层维护一个内部状态，在序列处理中传递信息。
 * <p>
 * RNN 公式:
 * h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
 * <p>
 * 其中:
 * - h_t 是当前时间步的隐藏状态
 * - x_t 是当前时间步的输入
 * - h_{t-1} 是前一个时间步的隐藏状态
 * - W_xh 是输入到隐藏状态的权重矩阵
 * - W_hh 是隐藏状态到隐藏状态的权重矩阵
 * - b 是偏置项
 */
public class SimpleRnnLayer extends RnnLayer {

    /**
     * 输入到隐藏状态的权重矩阵参数
     * 形状: (input_size, hidden_size)
     */
    ParameterV1 x2h;

    /**
     * 隐藏状态到隐藏状态的权重矩阵参数
     * 形状: (hidden_size, hidden_size)
     */
    ParameterV1 h2h;

    /**
     * 偏置参数
     * 形状: (1, hidden_size)
     */
    ParameterV1 b;

    /**
     * 当前时间步的隐藏状态变量
     */
    private Variable state;

    /**
     * 当前时间步的隐藏状态值（NdArray形式）
     */
    private NdArray stateValue;

    // 用于反向传播的缓存变量
    /**
     * 前一个时间步的隐藏状态变量
     */
    private Variable prevState;

    /**
     * tanh激活函数之前的值，用于反向传播计算
     */
    private Variable preTanh;

    /**
     * 输入线性变换结果 (x * W_xh + b)
     */
    private Variable xLinear;

    /**
     * 隐藏状态线性变换结果 (h_{t-1} * W_hh)
     */
    private Variable hLinear;

    /**
     * 隐藏层大小
     */
    private int hiddenSize;

    /**
     * 当前状态的批大小
     * 用于检测批大小变化并重置状态
     */
    private int currentBatchSize = -1;


    public SimpleRnnLayer(String name) {
        super(name);
    }

    /**
     * 构造一个SimpleRnnLayer实例
     *
     * @param name         层名称
     * @param xInputShape  输入形状 (batch_size, input_size)
     * @param yOutputShape 输出形状 (batch_size, hidden_size)
     */
    public SimpleRnnLayer(String name, Shape xInputShape, Shape yOutputShape) {
        super(name, xInputShape, yOutputShape);
        this.hiddenSize = yOutputShape.getColumn();
        init();
    }

    /**
     * 重置RNN层的内部状态
     * 在处理新序列之前应调用此方法
     */
    @Override
    public void resetState() {
        state = null;
        stateValue = null;
        currentBatchSize = -1; // 重置批大小记录
    }

    /**
     * 初始化RNN层的参数
     * 包括输入到隐藏状态权重、隐藏状态到隐藏状态权重和偏置项
     */
    @Override
    public void init() {
        int inputSize = inputShape.getColumn();

        // 初始化输入到隐藏状态的权重矩阵，使用Xavier初始化
        NdArray initWeight = NdArray.likeRandomN(Shape.of(inputSize, hiddenSize))
                .mulNum(Math.sqrt((double) 1 / inputSize));
        x2h = new ParameterV1(initWeight);
        x2h.setName(getName() + ".x2h");
        addParam(x2h.getName(), x2h);

        // 初始化隐藏状态到隐藏状态的权重矩阵，使用Xavier初始化
        initWeight = NdArray.likeRandomN(Shape.of(hiddenSize, hiddenSize)).mulNum(Math.sqrt((double) 1 / hiddenSize));
        h2h = new ParameterV1(initWeight);
        h2h.setName(getName() + ".h2h");
        addParam(h2h.getName(), h2h);

        // 初始化偏置项为零
        b = new ParameterV1(NdArray.zeros(Shape.of(1, hiddenSize)));
        b.setName(getName() + ".b");
        addParam(b.getName(), b);
    }

    /**
     * 基于Variable的前向传播方法
     * 支持动态批大小处理
     *
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 当前时间步的隐藏状态
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        int inputBatchSize = x.getValue().getShape().getRow();

        // 检测批大小变化，如果变化则重置状态
        if (currentBatchSize != -1 && currentBatchSize != inputBatchSize) {
            // 批大小变化，重置状态以适应新的批大小
            resetState();
        }
        currentBatchSize = inputBatchSize;

        // 第一次前向传播，没有前一时间步的隐藏状态
        if (Objects.isNull(state)) {
            prevState = null;
            xLinear = x.linear(x2h, b);
            state = xLinear.tanh();
            stateValue = state.getValue();
            preTanh = state;
        } else {
            // 后续前向传播，包含前一时间步的隐藏状态
            // 检查状态形状是否与当前输入匹配
            if (state.getValue().getShape().getRow() != inputBatchSize) {
                // 状态批大小不匹配，重新初始化状态
                resetState();
                return layerForward(inputs); // 递归调用处理重置后的状态
            }

            prevState = state;
            xLinear = x.linear(x2h, b);
            hLinear = new Variable(stateValue).linear(h2h, null);
            state = xLinear.add(hLinear).tanh();
            stateValue = state.getValue();
            preTanh = state;
        }
        return state;
    }


    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}