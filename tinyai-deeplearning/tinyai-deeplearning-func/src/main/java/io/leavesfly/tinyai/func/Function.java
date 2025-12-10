package io.leavesfly.tinyai.func;


import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 抽象的数学函数基类
 * <p>
 * 在TinyAI深度学习框架中，Function类是所有数学函数操作的基类。
 * 它定义了前向传播和反向传播的接口，并负责构建计算图。
 * 每个函数实例都维护输入变量和输出变量之间的关系。
 * 
 * <p><b>设计理念</b>：
 * <ul>
 *   <li>前向传播：将输入Variable转换为输出Variable</li>
 *   <li>反向传播：根据输出梯度计算输入梯度</li>
 *   <li>计算图构建：自动连接Variable和Function节点</li>
 * </ul>
 * 
 * @see Variable 自动微分变量
 */
public abstract class Function implements Serializable {

    // =============================================================================
    // 常量定义
    // =============================================================================
    
    /**
     * 标识函数可接受任意数量输入参数的特殊值
     */
    protected static final int ARBITRARY_INPUT_NUM = -1;

    // =============================================================================
    // 核心字段 - 计算图节点信息
    // =============================================================================

    /**
     * 函数的输入变量数组
     * <p>
     * 存储传递给该函数的所有输入变量，用于反向传播时追溯计算图
     */
    protected Variable[] inputs;

    /**
     * 函数的输出变量(单输出场景)
     * <p>
     * 存储该函数计算结果的输出变量。对于多输出函数，这是outputs[0]的引用
     */
    protected Variable output;

    /**
     * 函数的输出变量数组(多输出场景)
     * <p>
     * 对于单输出函数，该数组长度为1；对于多输出函数(如split)，包含所有输出
     */
    protected Variable[] outputs;

    // =============================================================================
    // 函数调用入口 - 前向传播执行
    // =============================================================================

    /**
     * 单输出函数的调用入口
     * <p>
     * 该方法是所有单输出函数的统一执行入口，完成以下流程：
     * <ol>
     *   <li>验证输入变量数量是否符合要求</li>
     *   <li>验证输入变量是否为null</li>
     *   <li>提取输入变量的NdArray值</li>
     *   <li>调用forward()执行具体的前向传播计算</li>
     *   <li>将计算结果包装为Variable</li>
     *   <li>在训练模式下构建计算图(设置creator)</li>
     * </ol>
     * 
     * <p><b>使用示例</b>：
     * <pre>{@code
     * Variable x = new Variable(ndArray);
     * Variable y = new Add().call(x, new Variable(1.0));
     * }</pre>
     *
     * @param _inputs 输入变量数组(可变参数)
     * @return 计算结果的输出变量
     * @throws RuntimeException 当输入变量数量不符合要求或包含null时抛出异常
     */
    public Variable call(Variable... _inputs) {
        // 步骤1: 验证输入参数
        validateInputs(_inputs);

        // 步骤2: 提取NdArray值
        NdArray[] ndArrayInputs = extractNdArrays(_inputs);

        // 步骤3: 执行前向传播
        NdArray ndArrayOutput = forward(ndArrayInputs);

        // 步骤4: 创建输出变量
        Variable _output = new Variable(ndArrayOutput);

        // 步骤5: 构建计算图(仅在训练模式且需要梯度时)
        if (shouldBuildGraph(_inputs)) {
            buildComputationGraph(_inputs, _output);
        }

        return _output;
    }

    /**
     * 多输出函数的调用入口
     * <p>
     * 用于处理返回多个输出的函数(如split, chunk等)。执行流程与call()类似，
     * 但支持返回多个Variable输出。
     * 
     * <p><b>使用示例</b>：
     * <pre>{@code
     * Variable x = new Variable(ndArray);
     * Variable[] outputs = new Split(2).callMulti(x);
     * }</pre>
     *
     * @param _inputs 输入变量数组(可变参数)
     * @return 输出变量数组
     * @throws RuntimeException 当输入变量数量不符合要求或包含null时抛出异常
     */
    public Variable[] callMulti(Variable... _inputs) {
        // 步骤1: 验证输入参数
        validateInputs(_inputs);

        // 步骤2: 提取NdArray值
        NdArray[] ndArrayInputs = extractNdArrays(_inputs);

        // 步骤3: 执行前向传播(多输出版本)
        NdArray[] ndArrayOutputs = forwardMulti(ndArrayInputs);

        // 步骤4: 创建输出变量数组
        Variable[] _outputs = Arrays.stream(ndArrayOutputs)
                .map(Variable::new)
                .toArray(Variable[]::new);

        // 步骤5: 构建计算图(仅在训练模式且需要梯度时)
        if (shouldBuildGraph(_inputs)) {
            buildComputationGraphMulti(_inputs, _outputs);
        }

        return _outputs;
    }

    // =============================================================================
    // 抽象方法 - 子类必须实现
    // =============================================================================

    /**
     * 前向传播计算(抽象方法)
     * <p>
     * 子类必须实现此方法来定义具体的前向传播计算逻辑。
     * 该方法在NdArray层面进行计算，不涉及Variable和计算图。
     * 
     * <p><b>实现要求</b>：
     * <ul>
     *   <li>纯函数：相同输入必须产生相同输出</li>
     *   <li>无副作用：不应修改输入数组</li>
     *   <li>返回新数组：计算结果应为新创建的NdArray</li>
     * </ul>
     *
     * @param inputs 输入的NdArray数组
     * @return 前向传播计算结果的NdArray(新创建)
     */
    public abstract NdArray forward(NdArray... inputs);

    /**
     * 前向传播计算 - 多输出版本(可选实现)
     * <p>
     * 对于返回多个输出的函数(如split, chunk)，子类应重写此方法。
     * 默认实现抛出UnsupportedOperationException。
     * 
     * @param inputs 输入的NdArray数组
     * @return 前向传播计算结果的NdArray数组
     * @throws UnsupportedOperationException 当函数不支持多输出时抛出
     */
    public NdArray[] forwardMulti(NdArray... inputs) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not support multiple outputs");
    }

    /**
     * 反向传播计算(抽象方法)
     * <p>
     * 子类必须实现此方法来定义具体的反向传播计算逻辑。
     * 根据链式法则，该方法根据输出梯度计算输入梯度。
     * 
     * <p><b>数学原理</b>：
     * 对于函数 y = f(x₁, x₂, ..., xₙ)，如果已知 ∂L/∂y，则：
     * <pre>
     * ∂L/∂x₁ = ∂L/∂y · ∂y/∂x₁
     * ∂L/∂x₂ = ∂L/∂y · ∂y/∂x₂
     * ...
     * </pre>
     * 
     * <p><b>实现要求</b>：
     * <ul>
     *   <li>返回列表长度必须等于输入数量(与requireInputNum()一致)</li>
     *   <li>如果某个输入不可导，对应梯度可以为null</li>
     *   <li>梯度形状必须与对应输入形状一致</li>
     * </ul>
     *
     * @param yGrad 输出变量的梯度 ∂L/∂y
     * @return 输入变量的梯度列表 [∂L/∂x₁, ∂L/∂x₂, ...]
     */
    public abstract List<NdArray> backward(NdArray yGrad);

    /**
     * 反向传播计算 - 多输出版本(可选实现)
     * <p>
     * 对于返回多个输出的函数(如split)，子类应重写此方法。
     * 接收所有输出的梯度，计算输入的梯度。
     * 
     * @param yGrads 与输出一一对应的梯度列表 [∂L/∂y₁, ∂L/∂y₂, ...]
     * @return 输入变量的梯度列表 [∂L/∂x₁, ∂L/∂x₂, ...]
     * @throws UnsupportedOperationException 当函数不支持多输出时抛出
     */
    public List<NdArray> backwardMulti(List<NdArray> yGrads) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not support multiple outputs backward");
    }

    /**
     * 获取函数所需的输入参数个数(抽象方法)
     * <p>
     * 子类实现此方法来指定函数所需的输入变量数量。
     * 
     * <p><b>返回值说明</b>：
     * <ul>
     *   <li>正整数：固定输入数量，如Add需要2个输入</li>
     *   <li>{@link #ARBITRARY_INPUT_NUM}(-1)：可变输入数量，如Concat</li>
     * </ul>
     *
     * @return 函数所需的输入参数个数，-1表示任意数量
     */
    public abstract int requireInputNum();

    // =============================================================================
    // Getter/Setter 方法
    // =============================================================================

    /**
     * 获取函数的输入变量数组
     * <p>
     * 用于反向传播时访问输入变量
     *
     * @return 输入变量数组，如果未构建计算图则为null
     */
    public Variable[] getInputs() {
        return inputs;
    }

    /**
     * 设置函数的输入变量数组
     * <p>
     * 通常由框架内部调用，用户代码一般不需要调用此方法
     *
     * @param inputs 输入变量数组
     */
    public void setInputs(Variable[] inputs) {
        this.inputs = inputs;
    }

    /**
     * 获取函数的输出变量(单输出)
     * <p>
     * 对于多输出函数，返回第一个输出
     *
     * @return 输出变量，如果未构建计算图则为null
     */
    public Variable getOutput() {
        return output;
    }

    /**
     * 获取函数的输出变量数组(多输出)
     * 
     * @return 输出变量数组，如果未构建计算图则为null
     */
    public Variable[] getOutputs() {
        return outputs;
    }

    /**
     * 设置函数的输出变量
     * <p>
     * 通常由框架内部调用，用户代码一般不需要调用此方法
     *
     * @param output 输出变量
     */
    public void setOutput(Variable output) {
        this.output = output;
    }

    /**
     * 设置函数的输出变量数组
     * <p>
     * 通常由框架内部调用，用户代码一般不需要调用此方法
     *
     * @param outputs 输出变量数组
     */
    public void setOutputs(Variable[] outputs) {
        this.outputs = outputs;
    }

    // =============================================================================
    // 工具方法 - 内部辅助功能
    // =============================================================================

    /**
     * 清理函数资源，断开计算图连接
     * <p>
     * 用于RNN/LSTM等循环网络中切断计算图，防止梯度回传过长。
     * 调用此方法后，该函数节点将从计算图中移除。
     * 
     * <p><b>使用场景</b>：
     * <ul>
     *   <li>截断式反向传播(TBPTT)</li>
     *   <li>内存优化：及时释放不再需要的计算图节点</li>
     * </ul>
     */
    public void unChain() {
        this.inputs = null;
        this.output = null;
        this.outputs = null;
    }

    /**
     * 判断当前函数是否为多输出函数
     * 
     * @return 如果函数返回多个输出则为true，否则为false
     */
    public boolean isMultiOutput() {
        return outputs != null && outputs.length > 1;
    }

    /**
     * 判断是否需要构建计算图
     * <p>
     * 只有在以下条件都满足时才构建计算图：
     * <ol>
     *   <li>当前处于训练模式(Config.train = true)</li>
     *   <li>至少有一个输入变量需要计算梯度(requireGrad = true)</li>
     * </ol>
     * 
     * @param vars 输入变量数组
     * @return 如果需要构建计算图则为true，否则为false
     */
    protected boolean shouldBuildGraph(Variable[] vars) {
        if (!Config.train) {
            return false;
        }
        return Arrays.stream(vars).anyMatch(v -> v != null && v.isRequireGrad());
    }

    /**
     * 验证输入变量的合法性
     * <p>
     * 检查输入数量和null值
     * 
     * @param inputs 输入变量数组
     * @throws RuntimeException 当输入不合法时抛出
     */
    private void validateInputs(Variable[] inputs) {
        int required = requireInputNum();
        if (required >= 0 && inputs.length != required) {
            throw new RuntimeException(
                    String.format("%s requires %d inputs, but got %d",
                            this.getClass().getSimpleName(), required, inputs.length));
        }
        if (Arrays.stream(inputs).anyMatch(Objects::isNull)) {
            throw new RuntimeException(
                    this.getClass().getSimpleName() + " inputs cannot contain null");
        }
    }

    /**
     * 从Variable数组中提取NdArray值
     * 
     * @param vars Variable数组
     * @return NdArray数组
     */
    private NdArray[] extractNdArrays(Variable[] vars) {
        return Arrays.stream(vars)
                .map(Variable::getValue)
                .toArray(NdArray[]::new);
    }

    /**
     * 构建计算图(单输出版本)
     * <p>
     * 设置输入输出关系并连接计算图节点
     * 
     * @param inputs 输入变量数组
     * @param output 输出变量
     */
    private void buildComputationGraph(Variable[] inputs, Variable output) {
        this.inputs = inputs;
        this.output = output;
        this.outputs = new Variable[]{output};
        output.setCreator(this);
    }

    /**
     * 构建计算图(多输出版本)
     * <p>
     * 设置输入输出关系并连接计算图节点
     * 
     * @param inputs 输入变量数组
     * @param outputs 输出变量数组
     */
    private void buildComputationGraphMulti(Variable[] inputs, Variable[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.output = outputs.length > 0 ? outputs[0] : null;
        for (Variable out : outputs) {
            out.setCreator(this);
        }
    }
}