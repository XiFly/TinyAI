package io.leavesfly.tinyai.nnet.v2.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

/**
 * V2版本的Conv2d层
 * <p>
 * 二维卷积层，用于处理图像等二维数据。
 * <p>
 * 实现标准的卷积操作，使用Im2Col技术将卷积转换为矩阵乘法。
 * <p>
 * 公式：
 * output = Conv2d(input, weight) + bias
 * <p>
 * 其中：
 * - input: (batch_size, in_channels, height, width)
 * - weight: (out_channels, in_channels, kernel_height, kernel_width)
 * - bias: (out_channels,)
 * - output: (batch_size, out_channels, out_height, out_width)
 * <p>
 * 输出尺寸计算：
 * out_height = (height + 2*padding - kernel_height) / stride + 1
 * out_width = (width + 2*padding - kernel_width) / stride + 1
 *
 * @author leavesfly
 * @version 2.0
 */
public class Conv2d extends Module {

    private Parameter weight;  // 卷积核权重
    private Parameter bias;    // 偏置（可选）

    private final int inChannels;   // 输入通道数
    private final int outChannels;  // 输出通道数
    private final int kernelHeight; // 卷积核高度
    private final int kernelWidth;  // 卷积核宽度
    private final int stride;       // 步长
    private final int padding;      // 填充
    private final boolean useBias;  // 是否使用偏置

    /**
     * 构造函数（正方形卷积核）
     *
     * @param name        层名称
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     * @param stride      步长
     * @param padding     填充
     * @param useBias     是否使用偏置
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelSize,
                  int stride, int padding, boolean useBias) {
        this(name, inChannels, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }

    /**
     * 构造函数（非正方形卷积核）
     *
     * @param name         层名称
     * @param inChannels   输入通道数
     * @param outChannels  输出通道数
     * @param kernelHeight 卷积核高度
     * @param kernelWidth  卷积核宽度
     * @param stride       步长
     * @param padding      填充
     * @param useBias      是否使用偏置
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
                  int stride, int padding, boolean useBias) {
        super(name);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;

        initializeParameters();
        init();
    }

    /**
     * 构造函数（默认参数）
     *
     * @param name        层名称
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelSize) {
        this(name, inChannels, outChannels, kernelSize, 1, 0, true);
    }

    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 权重形状: (out_channels, in_channels, kernel_height, kernel_width)
        Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
        weight = registerParameter("weight", new Parameter(NdArray.of(weightShape)));

        if (useBias) {
            // 偏置形状: (out_channels,)
            bias = registerParameter("bias", new Parameter(NdArray.of(Shape.of(outChannels))));
        }
    }

    @Override
    public void resetParameters() {
        // 使用Kaiming初始化（He初始化）
        // 卷积层适合使用ReLU激活函数
        Initializers.kaimingUniform(weight.data());

        if (useBias) {
            // 偏置初始化为0
            Initializers.zeros(bias.data());
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();

        // 检查输入形状
        int[] dims = inputData.getShape().getShapeDims();
        if (dims.length != 4) {
            throw new IllegalArgumentException(
                    String.format("Expected 4D input (batch, channels, height, width), but got %dD", dims.length));
        }

        int batchSize = dims[0];
        int inputChannels = dims[1];
        int inputHeight = dims[2];
        int inputWidth = dims[3];

        if (inputChannels != inChannels) {
            throw new IllegalArgumentException(
                    String.format("Expected %d input channels, but got %d", inChannels, inputChannels));
        }

        // 计算输出尺寸
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;

        // 执行Im2Col转换
        NdArray im2colResult = performIm2Col(inputData, batchSize, inputChannels, 
                                             inputHeight, inputWidth, outputHeight, outputWidth);

        // 重塑权重为二维矩阵
        NdArray weightReshaped = reshapeWeight();

        // 矩阵乘法计算卷积
        Variable im2colVar = new Variable(im2colResult);
        Variable weightVar = new Variable(weightReshaped.transpose());
        Variable output = im2colVar.matMul(weightVar);

        // 添加偏置
        if (useBias) {
            output = addBias(output);
        }

        // 重塑输出为4维
        Shape outputShape = Shape.of(batchSize, outChannels, outputHeight, outputWidth);
        output = output.reshape(outputShape);

        return output;
    }

    /**
     * 执行Im2Col转换
     * <p>
     * 将卷积窗口展开为列，方便进行矩阵乘法
     */
    private NdArray performIm2Col(NdArray inputData, int batchSize, int channels,
                                   int height, int width, int outHeight, int outWidth) {
        int outputRows = batchSize * outHeight * outWidth;
        int outputCols = channels * kernelHeight * kernelWidth;

        float[] outputData = new float[outputRows * outputCols];

        int outputRowIndex = 0;
        for (int n = 0; n < batchSize; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int colIndex = 0;

                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < kernelHeight; fh++) {
                            int imRow = h * stride + fh - padding;
                            for (int fw = 0; fw < kernelWidth; fw++) {
                                int imCol = w * stride + fw - padding;

                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    outputData[outputRowIndex * outputCols + colIndex] =
                                            inputData.get(n, c, imRow, imCol);
                                } else {
                                    // 填充0
                                    outputData[outputRowIndex * outputCols + colIndex] = 0.0f;
                                }
                                colIndex++;
                            }
                        }
                    }
                    outputRowIndex++;
                }
            }
        }

        Shape outputShape = Shape.of(outputRows, outputCols);
        return NdArray.of(outputData, outputShape);
    }

    /**
     * 重塑权重为二维矩阵
     * <p>
     * 从 (out_channels, in_channels, kernel_h, kernel_w)
     * 到 (out_channels, in_channels * kernel_h * kernel_w)
     */
    private NdArray reshapeWeight() {
        NdArray weightData = weight.data();
        Shape newShape = Shape.of(outChannels, inChannels * kernelHeight * kernelWidth);
        return weightData.reshape(newShape);
    }

    /**
     * 添加偏置
     */
    private Variable addBias(Variable output) {
        NdArray biasData = bias.data();
        NdArray outputData = output.getValue();

        float[][] outputMatrix = outputData.getMatrix();
        float[] biasArray = biasData.getArray();

        for (int i = 0; i < outputMatrix.length; i++) {
            for (int j = 0; j < outputMatrix[i].length; j++) {
                outputMatrix[i][j] += biasArray[j];
            }
        }

        return new Variable(NdArray.of(outputMatrix));
    }

    public int getInChannels() {
        return inChannels;
    }

    public int getOutChannels() {
        return outChannels;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    @Override
    public String toString() {
        return "Conv2d{" +
                "name='" + name + '\'' +
                ", inChannels=" + inChannels +
                ", outChannels=" + outChannels +
                ", kernelSize=(" + kernelHeight + ", " + kernelWidth + ")" +
                ", stride=" + stride +
                ", padding=" + padding +
                ", useBias=" + useBias +
                '}';
    }
}
