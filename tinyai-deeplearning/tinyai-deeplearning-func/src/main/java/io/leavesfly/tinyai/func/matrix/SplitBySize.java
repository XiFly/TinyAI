package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * SplitBySize算子 - 按指定大小在维度上分割张量
 * <p>
 * 类似 PyTorch 的 torch.split(tensor, split_size, dim)
 * 将张量在指定维度上分割成多个块,每个块的大小为split_size(最后一块可能更小)
 * 
 * <p>示例:
 * <pre>
 * 输入: [2, 5] 的张量
 * split(splitSize=2, dim=1) -> [[2, 2], [2, 2], [2, 1]] 三个张量
 * </pre>
 * 
 * @author TinyAI
 */
public class SplitBySize extends Function {

    private final int splitSize;
    private final int dim;
    private Shape inputShape;
    private int actualDim;
    private int[] splitSizes;  // 每个分块的实际大小

    /**
     * 构造函数
     *
     * @param splitSize 每个分块的大小
     * @param dim      分割的维度索引(支持负数)
     */
    public SplitBySize(int splitSize, int dim) {
        this.splitSize = splitSize;
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        throw new UnsupportedOperationException(
                "SplitBySize is a multi-output function, use forwardMulti instead");
    }

    @Override
    public NdArray[] forwardMulti(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();
        int ndim = inputDims.length;

        // 处理负数维度
        actualDim = dim < 0 ? ndim + dim : dim;
        if (actualDim < 0 || actualDim >= ndim) {
            throw new IllegalArgumentException(
                    String.format("SplitBySize: dimension out of range: %d for shape %s", 
                            dim, inputShape));
        }

        int dimSize = inputDims[actualDim];
        
        // 计算分块数量
        int numSplits = (dimSize + splitSize - 1) / splitSize;  // 向上取整
        splitSizes = new int[numSplits];
        
        // 计算每个分块的实际大小
        for (int i = 0; i < numSplits; i++) {
            int start = i * splitSize;
            int end = Math.min(start + splitSize, dimSize);
            splitSizes[i] = end - start;
        }

        // 执行分割
        NdArray[] outputs = new NdArray[numSplits];
        for (int i = 0; i < numSplits; i++) {
            int start = i * splitSize;
            int end = Math.min(start + splitSize, dimSize);
            outputs[i] = sliceAlongDim(x, actualDim, start, end);
        }

        return outputs;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        throw new UnsupportedOperationException(
                "SplitBySize is a multi-output function, use backwardMulti instead");
    }

    @Override
    public List<NdArray> backwardMulti(List<NdArray> yGrads) {
        // Split的反向传播是将所有输出的梯度拼接回输入形状
        
        int[] inputDims = inputShape.getShapeDims();
        float[] gradData = new float[inputShape.size()];
        
        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        
        // 遍历每个分块的梯度
        int currentOffset = 0;
        for (int splitIdx = 0; splitIdx < yGrads.size(); splitIdx++) {
            NdArray splitGrad = yGrads.get(splitIdx);
            if (splitGrad == null) {
                currentOffset += splitSizes[splitIdx];
                continue;
            }
            
            float[] splitGradData = splitGrad.getArray();
            int[] splitGradDims = splitGrad.getShape().getShapeDims();
            
            // 遍历该分块的所有元素
            int[] splitIdx_arr = new int[splitGradDims.length];
            for (int i = 0; i < splitGradData.length; i++) {
                // 计算分块内索引
                int pos = i;
                for (int d = splitGradDims.length - 1; d >= 0; d--) {
                    splitIdx_arr[d] = pos % splitGradDims[d];
                    pos /= splitGradDims[d];
                }
                
                // 计算对应的输入索引(在分割维度上加上偏移)
                int[] inputIdx = splitIdx_arr.clone();
                inputIdx[actualDim] = splitIdx_arr[actualDim] + currentOffset;
                
                // 计算输入位置
                int inputPos = 0;
                for (int d = 0; d < inputDims.length; d++) {
                    inputPos += inputIdx[d] * inputStrides[d];
                }
                
                gradData[inputPos] = splitGradData[i];
            }
            
            currentOffset += splitSizes[splitIdx];
        }
        
        List<NdArray> result = new ArrayList<>();
        result.add(NdArray.of(gradData, inputShape));
        return result;
    }

    /**
     * 沿指定维度进行范围切片
     */
    private NdArray sliceAlongDim(NdArray x, int dim, int start, int end) {
        float[] xData = x.getArray();
        int[] inputDims = inputShape.getShapeDims();
        
        // 计算输出形状
        int[] outputDims = inputDims.clone();
        outputDims[dim] = end - start;
        Shape outputShape = Shape.of(outputDims);
        
        float[] outputData = new float[outputShape.size()];
        
        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        
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
