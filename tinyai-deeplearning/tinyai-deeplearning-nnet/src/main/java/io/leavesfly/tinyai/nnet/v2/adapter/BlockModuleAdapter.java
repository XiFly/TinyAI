package io.leavesfly.tinyai.nnet.v2.adapter;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.ParameterV1;
import io.leavesfly.tinyai.nnet.v2.core.Module;

import java.util.Map;

/**
 * V1 Block 到 V2 Module 的适配器
 * <p>
 * 将 V1 版本的 Block 适配为 V2 版本的 Module，使 Model 类能够使用 V1 的 Block。
 * 这个适配器实现了 Module 接口，内部委托给 Block 执行实际操作。
 * <p>
 * 使用场景：
 * - 当需要将现有的 V1 Block 用于支持 V2 Module 的 Model 时
 * - 保持向后兼容性，允许旧代码继续工作
 *
 * @author TinyAI
 * @version 2.0
 */
public class BlockModuleAdapter extends Module {

    /**
     * 被适配的 V1 Block 实例
     */
    private final Block block;

    /**
     * 缓存的输入形状（用于性能优化）
     */
    private Shape cachedInputShape;

    /**
     * 缓存的输出形状（用于性能优化）
     */
    private Shape cachedOutputShape;

    /**
     * 构造函数
     *
     * @param block 要适配的 V1 Block 实例
     */
    public BlockModuleAdapter(Block block) {
        super(block != null ? block.getName() : "BlockAdapter");
        this.block = block;
        if (block != null) {
            this.cachedInputShape = block.getInputShape();
            this.cachedOutputShape = block.getOutputShape();
            // 初始化 Block
            if (!block.isAlreadyInit()) {
                block.init();
            }
            // 将 Block 的参数注册到 Module 中
            registerBlockParameters();
        }
    }

    /**
     * 注册 Block 的所有参数到 Module 中
     * 将 V1 的 ParameterV1 转换为 V2 的 ParameterV1
     */
    private void registerBlockParameters() {
        if (block == null) {
            return;
        }

        Map<String, ParameterV1> blockParams = block.getAllParams();
        for (Map.Entry<String, ParameterV1> entry : blockParams.entrySet()) {
            String paramName = entry.getKey();
            ParameterV1 v1Param = entry.getValue();
            
            if (v1Param != null) {
                // 将 V1 ParameterV1 转换为 V2 ParameterV1
                // V1 ParameterV1 通过 isRequireGrad() 方法获取是否需要梯度
                boolean requiresGrad = v1Param.isRequireGrad();
                
                io.leavesfly.tinyai.nnet.v2.core.Parameter v2Param = 
                    new io.leavesfly.tinyai.nnet.v2.core.Parameter(v1Param.getValue(), requiresGrad);
                // 如果 V1 ParameterV1 有梯度，复制梯度
                if (v1Param.getGrad() != null) {
                    v2Param.setGrad(v1Param.getGrad());
                }
                registerParameter(paramName, v2Param);
            }
        }
    }

    /**
     * Module 的前向传播方法
     * 委托给 Block 的 layerForward 方法
     *
     * @param inputs 输入变量数组
     * @return 前向传播结果
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (block == null) {
            throw new IllegalStateException("Block is null, cannot perform forward");
        }
        return block.layerForward(inputs);
    }

    /**
     * 获取所有参数（V2 接口）
     * 返回 V2 格式的参数映射
     *
     * @return 参数映射（使用 V2 ParameterV1）
     */
    @Override
    public Map<String, io.leavesfly.tinyai.nnet.v2.core.Parameter> namedParameters() {
        // 同步 Block 的参数到 Module
        syncParametersFromBlock();
        return super.namedParameters();
    }

    /**
     * 从 Block 同步参数到 Module
     * 确保参数值是最新的
     */
    private void syncParametersFromBlock() {
        if (block == null) {
            return;
        }

        Map<String, ParameterV1> blockParams = block.getAllParams();
        Map<String, io.leavesfly.tinyai.nnet.v2.core.Parameter> moduleParams = _parameters;

        // 更新已存在的参数值
        for (Map.Entry<String, ParameterV1> entry : blockParams.entrySet()) {
            String paramName = entry.getKey();
            ParameterV1 v1Param = entry.getValue();
            
            if (v1Param != null && moduleParams.containsKey(paramName)) {
                io.leavesfly.tinyai.nnet.v2.core.Parameter v2Param = moduleParams.get(paramName);
                if (v2Param != null) {
                    // 更新参数值
                    v2Param.setValue(v1Param.getValue());
                    // 更新梯度
                    if (v1Param.getGrad() != null) {
                        v2Param.setGrad(v1Param.getGrad());
                    }
                }
            }
        }
    }

    /**
     * 清除所有参数的梯度
     * 同时清除 Block 和 Module 的梯度
     */
    @Override
    public void clearGrads() {
        super.clearGrads();
        if (block != null) {
            block.clearGrads();
        }
    }

    /**
     * 获取被适配的 Block 实例
     *
     * @return Block 实例
     */
    public Block getBlock() {
        return block;
    }

    /**
     * 获取输入形状
     * V2 Module 没有此方法，但为了兼容性提供
     *
     * @return 输入形状，如果未设置则返回 null
     */
    public Shape getInputShape() {
        if (cachedInputShape != null) {
            return cachedInputShape;
        }
        if (block != null) {
            return block.getInputShape();
        }
        return null;
    }

    /**
     * 获取输出形状
     * V2 Module 没有此方法，但为了兼容性提供
     *
     * @return 输出形状，如果未设置则返回 null
     */
    public Shape getOutputShape() {
        if (cachedOutputShape != null) {
            return cachedOutputShape;
        }
        if (block != null) {
            return block.getOutputShape();
        }
        return null;
    }

    /**
     * 重置模型状态
     * V2 Module 没有此方法，但为了兼容性提供
     * 主要用于 RNN 等有状态的模型
     */
    public void resetState() {
        if (block != null) {
            block.resetState();
        }
    }

    /**
     * 初始化方法
     * 确保 Block 已初始化
     */
    @Override
    public void init() {
        super.init();
        if (block != null && !block.isAlreadyInit()) {
            block.init();
        }
    }

    @Override
    public String toString() {
        return "BlockModuleAdapter{" +
                "block=" + (block != null ? block.getName() : "null") +
                ", name='" + name + '\'' +
                '}';
    }
}

