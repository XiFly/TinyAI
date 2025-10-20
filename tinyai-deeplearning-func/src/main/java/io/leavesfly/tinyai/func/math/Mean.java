package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 均值函数
 * <p>
 * 计算输入数组沿指定轴的均值。
 *
 * @author leavesfly
 * @version 1.0
 */
public class Mean extends Function {

    private int axis;
    private boolean keepdims;
    private Shape inputShape;
    private int axisSize;

    /**
     * 构造函数
     *
     * @param axis     指定轴
     * @param keepdims 是否保持维度
     */
    public Mean(int axis, boolean keepdims) {
        this.axis = axis;
        this.keepdims = keepdims;
    }

    /**
     * 前向传播计算均值
     * <p>
     * 计算输入数组沿指定轴的均值。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 均值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        int[] shape = inputShape.getShapeDims();
        
        // 处理负数轴索引
        int actualAxis = axis;
        if (actualAxis < 0) {
            actualAxis = shape.length + actualAxis;
        }
        
        // 保存轴的大小，用于反向传播
        axisSize = shape[actualAxis];
        
        NdArray result = inputs[0].mean(actualAxis);
        
        if (keepdims) {
            return result.broadcastTo(inputShape);
        }
        return result;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于均值函数，梯度计算规则为：
     * - 将梯度值平均分配给所有元素
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        if (!keepdims) {
            yGrad = yGrad.broadcastTo(inputShape);
        }
        // 梯度需要除以轴的大小，因为是平均值
        NdArray grad = yGrad.divNum(axisSize);
        return Collections.singletonList(grad);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 均值函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
