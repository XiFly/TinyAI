package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 方差函数
 * <p>
 * 计算输入数组沿指定轴的方差。
 * <p>
 * 方差计算公式：Var(X) = E[(X - E[X])^2]
 *
 * @author leavesfly
 * @version 1.0
 */
public class Variance extends Function {

    private int axis;
    private boolean keepdims;
    private Shape inputShape;
    private int axisSize;
    private NdArray mean;

    /**
     * 构造函数
     *
     * @param axis     指定轴
     * @param keepdims 是否保持维度
     */
    public Variance(int axis, boolean keepdims) {
        this.axis = axis;
        this.keepdims = keepdims;
    }

    /**
     * 前向传播计算方差
     * <p>
     * 计算输入数组沿指定轴的方差。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 方差的NdArray
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
        
        NdArray result = inputs[0].var(actualAxis);
        
        if (keepdims) {
            // 保存均值用于反向传播
            mean = inputs[0].mean(actualAxis).broadcastTo(inputShape);
            return result.broadcastTo(inputShape);
        } else {
            // 保存均值用于反向传播
            mean = inputs[0].mean(actualAxis);
        }
        return result;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于方差函数，梯度计算公式为：
     * d(Var)/dx = 2 * (x - mean) / n
     * 其中 n 是轴的大小
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        
        if (!keepdims) {
            yGrad = yGrad.broadcastTo(inputShape);
            mean = mean.broadcastTo(inputShape);
        }
        
        // 梯度计算: 2 * (x - mean) / n * yGrad
        NdArray grad = x.sub(mean).mulNum(2.0f / axisSize).mul(yGrad);
        return Collections.singletonList(grad);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 方差函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
