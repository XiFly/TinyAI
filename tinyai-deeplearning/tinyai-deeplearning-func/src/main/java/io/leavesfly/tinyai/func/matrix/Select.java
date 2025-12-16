package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Select算子 - 在指定维度选择单个索引
 * <p>
 * 类似 PyTorch 的 x.select(dim, index)
 * 在指定维度上选择单个索引,该维度会被移除
 * 
 * <p>示例:
 * <pre>
 * 输入: [2, 3, 4] 的张量
 * select(dim=1, index=0) -> [2, 4] 的张量
 * </pre>
 * 
 * @author TinyAI
 */
public class Select extends Function {

    private final int dim;
    private final int index;
    private Shape inputShape;
    private int actualDim;

    /**
     * 构造函数
     *
     * @param dim   要选择的维度索引(支持负数)
     * @param index 要选择的索引位置
     */
    public Select(int dim, int index) {
        this.dim = dim;
        this.index = index;
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
                    String.format("Select: dimension out of range: %d for shape %s", 
                            dim, inputShape));
        }

        // 处理负数索引
        int actualIndex = index < 0 ? inputDims[actualDim] + index : index;
        if (actualIndex < 0 || actualIndex >= inputDims[actualDim]) {
            throw new IndexOutOfBoundsException(
                    String.format("Select: index %d out of range [0, %d) for dimension %d",
                            index, inputDims[actualDim], actualDim));
        }

        // 计算输出形状(移除选择的维度)
        List<Integer> outputDimsList = new ArrayList<>();
        for (int i = 0; i < ndim; i++) {
            if (i != actualDim) {
                outputDimsList.add(inputDims[i]);
            }
        }
        
        // 如果所有维度都被移除,保留一个维度为1
        if (outputDimsList.isEmpty()) {
            outputDimsList.add(1);
        }
        
        int[] outputDims = outputDimsList.stream().mapToInt(i -> i).toArray();
        Shape outputShape = Shape.of(outputDims);

        // 执行选择操作
        return selectAlongDim(x, actualDim, actualIndex, outputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度需要scatter回原始形状
        // 在被选择的维度上,只有选中的索引位置有梯度,其他位置为0
        
        int[] inputDims = inputShape.getShapeDims();
        float[] gradData = new float[inputShape.size()];
        float[] yGradData = yGrad.getArray();
        int[] yGradDims = yGrad.getShape().getShapeDims();

        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] yGradStrides = computeStrides(yGradDims);

        // 处理负数索引
        int actualIndex = index < 0 ? inputDims[actualDim] + index : index;

        // 遍历yGrad的所有元素,scatter回输入形状
        int[] yGradIdx = new int[yGradDims.length];
        for (int i = 0; i < yGradData.length; i++) {
            // 计算yGrad索引
            int pos = i;
            for (int d = yGradDims.length - 1; d >= 0; d--) {
                yGradIdx[d] = pos % yGradDims[d];
                pos /= yGradDims[d];
            }

            // 构造输入索引(插入被移除的维度)
            int[] inputIdx = new int[inputDims.length];
            int yGradDimIdx = 0;
            for (int d = 0; d < inputDims.length; d++) {
                if (d == actualDim) {
                    inputIdx[d] = actualIndex;
                } else {
                    inputIdx[d] = yGradIdx[yGradDimIdx++];
                }
            }

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
     * 沿指定维度选择元素
     */
    private NdArray selectAlongDim(NdArray x, int dim, int index, Shape outputShape) {
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

            // 构造输入索引(插入选择的维度)
            int[] inputIdx = new int[inputDims.length];
            int outputDimIdx = 0;
            for (int d = 0; d < inputDims.length; d++) {
                if (d == dim) {
                    inputIdx[d] = index;
                } else {
                    inputIdx[d] = outputIdx[outputDimIdx++];
                }
            }

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
