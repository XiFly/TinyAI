package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * SliceRange算子 - 在指定维度进行范围切片
 * <p>
 * 类似 PyTorch 的 x[start:end]
 * 在指定维度上选择一个范围的索引,保持该维度
 * 
 * <p>示例:
 * <pre>
 * 输入: [2, 5, 4] 的张量
 * sliceRange(dim=1, start=1, end=4) -> [2, 3, 4] 的张量
 * </pre>
 * 
 * @author TinyAI
 */
public class SliceRange extends Function {

    private final int dim;
    private final int start;
    private final int end;
    private Shape inputShape;
    private int actualDim;
    private int actualStart;
    private int actualEnd;

    /**
     * 构造函数
     *
     * @param dim   要切片的维度索引(支持负数)
     * @param start 起始索引(包含,支持负数)
     * @param end   结束索引(不包含,支持负数)
     */
    public SliceRange(int dim, int start, int end) {
        this.dim = dim;
        this.start = start;
        this.end = end;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();
        int ndim = inputDims.length;

        // 处理负数维度
        actualDim = dim < 0 ? ndim + dim : dim;
        if (actualDim < 0 || actualDim >= ndim) {
            throw new IllegalArgumentException(
                    String.format("SliceRange: dimension out of range: %d for shape %s", 
                            dim, inputShape));
        }

        int dimSize = inputDims[actualDim];

        // 处理负数索引
        actualStart = start < 0 ? dimSize + start : start;
        actualEnd = end < 0 ? dimSize + end : end;

        // 边界检查
        if (actualStart < 0) actualStart = 0;
        if (actualEnd > dimSize) actualEnd = dimSize;
        if (actualStart > actualEnd) actualStart = actualEnd;

        // 计算输出形状
        int[] outputDims = inputDims.clone();
        outputDims[actualDim] = actualEnd - actualStart;
        Shape outputShape = Shape.of(outputDims);

        // 执行切片操作
        return sliceAlongDim(x, actualDim, actualStart, actualEnd, outputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度需要scatter回原始形状
        // 在切片范围内的位置有梯度,其他位置为0
        
        int[] inputDims = inputShape.getShapeDims();
        float[] gradData = new float[inputShape.size()];
        float[] yGradData = yGrad.getArray();
        int[] yGradDims = yGrad.getShape().getShapeDims();

        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] yGradStrides = computeStrides(yGradDims);

        // 遍历yGrad的所有元素,scatter回输入形状
        int[] yGradIdx = new int[yGradDims.length];
        for (int i = 0; i < yGradData.length; i++) {
            // 计算yGrad索引
            int pos = i;
            for (int d = yGradDims.length - 1; d >= 0; d--) {
                yGradIdx[d] = pos % yGradDims[d];
                pos /= yGradDims[d];
            }

            // 计算对应的输入索引(在切片维度上加上偏移)
            int[] inputIdx = yGradIdx.clone();
            inputIdx[actualDim] = yGradIdx[actualDim] + actualStart;

            // 计算输入位置
            int inputPos = 0;
            for (int d = 0; d < inputDims.length; d++) {
                inputPos += inputIdx[d] * inputStrides[d];
            }

            gradData[inputPos] = yGradData[i];
        }

        return Collections.singletonList(NdArray.of(gradData, inputShape));
    }

    /**
     * 沿指定维度进行范围切片
     */
    private NdArray sliceAlongDim(NdArray x, int dim, int start, int end, Shape outputShape) {
        float[] xData = x.getArray();
        float[] outputData = new float[outputShape.size()];
        int[] inputDims = inputShape.getShapeDims();
        int[] outputDims = outputShape.getShapeDims();

        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] outputStrides = computeStrides(outputDims);

        // 遍历所有输出位置
        int[] outputIdx = new int[outputDims.length];
        for (int i = 0; i < outputData.length; i++) {
            // 计算输出索引
            int pos = i;
            for (int d = outputDims.length - 1; d >= 0; d--) {
                outputIdx[d] = pos % outputDims[d];
                pos /= outputDims[d];
            }

            // 计算输入索引(在切片维度上加上偏移)
            int[] inputIdx = outputIdx.clone();
            inputIdx[dim] = outputIdx[dim] + start;

            // 计算输入位置
            int inputPos = 0;
            for (int d = 0; d < inputDims.length; d++) {
                inputPos += inputIdx[d] * inputStrides[d];
            }

            outputData[i] = xData[inputPos];
        }

        return NdArray.of(outputData, outputShape);
    }

    /**
     * 计算步长
     */
    private int[] computeStrides(int[] dims) {
        int[] strides = new int[dims.length];
        strides[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
