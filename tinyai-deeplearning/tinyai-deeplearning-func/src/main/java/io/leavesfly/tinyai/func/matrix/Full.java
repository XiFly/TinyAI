package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * Full算子 - 创建指定形状和值的张量
 * <p>
 * 类似 PyTorch 的 torch.full(shape, value)
 * 创建一个指定形状,所有元素都为指定值的张量
 * 
 * <p>示例:
 * <pre>
 * full(Shape.of(2, 3), 5.0f) -> [[5, 5, 5], [5, 5, 5]]
 * </pre>
 * 
 * @author TinyAI
 */
public class Full extends Function {

    private final Shape shape;
    private final float value;

    /**
     * 构造函数
     *
     * @param shape 目标形状
     * @param value 填充值
     */
    public Full(Shape shape, float value) {
        this.shape = shape;
        this.value = value;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        // Full操作不需要输入,直接创建新张量
        // 但为了与Function框架兼容,可以接受空输入或虚拟输入
        return NdArray.like(shape, value);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // Full操作不依赖输入,梯度为null或空列表
        // 如果有虚拟输入,返回零梯度
        return Collections.emptyList();
    }

    @Override
    public int requireInputNum() {
        // Full操作不需要输入,返回0
        return 0;
    }
}
