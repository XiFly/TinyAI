package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.ParameterV1;

import java.util.ArrayList;
import java.util.List;

/**
 * 卷积层实现类
 * <p>
 * 实现了标准的卷积操作，支持步长、填充、偏置等参数。
 * 使用Im2Col技术将卷积操作转换为矩阵乘法，提高计算效率。
 * <p>
 * 卷积公式：output = input * weight + bias
 * 其中 weight形状为 (out_channels, in_channels, kernel_height, kernel_width)
 */
public class ConvLayer extends Layer {

    private ParameterV1 weight;        // 卷积核参数
    private ParameterV1 bias;          // 偏置参数(可选)

    private int inChannels;          // 输入通道数
    private int outChannels;         // 输出通道数
    private int kernelHeight;        // 卷积核高度
    private int kernelWidth;         // 卷积核宽度
    private int stride;              // 步长
    private int padding;             // 填充
    private boolean useBias;        // 是否使用偏置

    /**
     * 前向传播中缓存的中间结果，供反向传播使用
     */
    private NdArray lastInput;       // 输入缓存
    private NdArray lastIm2col;      // im2col 展开结果
    private int lastOutHeight;
    private int lastOutWidth;
    private int lastBatchSize;

    public ConvLayer(String name) {
        super(name);
    }

    /**
     * 构造卷积层
     *
     * @param name        层名称
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸(正方形)
     * @param stride      步长
     * @param padding     填充
     * @param useBias     是否使用偏置
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelSize,
                     int stride, int padding, boolean useBias) {
        this(name, inChannels, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }

    /**
     * 构造卷积层(非正方形卷积核)
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
                     int stride, int padding, boolean useBias) {
        super(name, null, null);  // 输入输出形状将在运行时确定

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;

        init();
    }

    public ConvLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 从输入形状推断参数(默认值)
        if (_inputShape != null && _inputShape.size() == 4) {
            this.inChannels = _inputShape.getDimension(1);
            this.outChannels = 32;  // 默认输出通道数
        } else {
            this.inChannels = 1;
            this.outChannels = 32;
        }
        this.kernelHeight = 3;
        this.kernelWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.useBias = true;

        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重参数 (out_channels, in_channels, kernel_height, kernel_width)
            // 使用He初始化
            double fan_in = inChannels * kernelHeight * kernelWidth;
            double std = Math.sqrt(2.0 / fan_in);

            Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
            NdArray weightData = NdArray.likeRandomN(weightShape).mulNum(std);

            weight = new ParameterV1(weightData);
            weight.setName(name + "_weight");
            addParam("weight", weight);

            // 初始化偏置参数(如果使用)
            if (useBias) {
                bias = new ParameterV1(NdArray.zeros(Shape.of(outChannels)));
                bias.setName(name + "_bias");
                addParam("bias", bias);
            }

            alreadyInit = true;
        }
    }


    private Variable layerForward0(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();

        // 检查输入形状 (batch_size, channels, height, width)
        if (inputData.getShape().getDimNum() != 4) {
            throw new RuntimeException("卷积层输入必须是4维的: (batch_size, channels, height, width)");
        }

        int batchSize = inputData.getShape().getDimension(0);
        int inputChannels = inputData.getShape().getDimension(1);
        int inputHeight = inputData.getShape().getDimension(2);
        int inputWidth = inputData.getShape().getDimension(3);

        // 检查通道数匹配
        if (inputChannels != inChannels) {
            throw new RuntimeException("输入通道数不匹配: 期望" + inChannels + ", 实际" + inputChannels);
        }

        // 计算输出尺寸
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;

        // 进行Im2Col转换，并缓存供反向传播使用
        NdArray im2colResult = performIm2Col(inputData, kernelHeight, kernelWidth, stride, padding);
        lastInput = inputData;
        lastIm2col = im2colResult;
        lastOutHeight = outputHeight;
        lastOutWidth = outputWidth;
        lastBatchSize = batchSize;

        // 重塑权重为二维矩阵
        NdArray weightReshaped = reshapeWeight();

        // 矩阵乘法计算卷积
        // Im2Col结果形状为 [batch*out_h*out_w, in_channels*kernel_h*kernel_w]
        // 权重形状为 [out_channels, in_channels*kernel_h*kernel_w]
        // 需要 im2col × weight.T = [batch*out_h*out_w, in_channels*kernel_h*kernel_w] × [in_channels*kernel_h*kernel_w, out_channels]
        Variable im2colVar = new Variable(im2colResult);
        Variable weightVar = new Variable(weightReshaped.transpose());
        Variable output = im2colVar.matMul(weightVar);

        // 添加偏置(如果有)
        if (useBias) {
            // 将偏置加到每个输出通道上
            NdArray biasData = bias.getValue();
            NdArray outputData = output.getValue();

            // 重塑输出为 [batch*out_h*out_w, out_channels]
            // 然后对每个out_channels维度加上对应的偏置值
            float[][] outputMatrix = outputData.getMatrix();
            for (int i = 0; i < outputMatrix.length; i++) {
                for (int j = 0; j < outputMatrix[i].length; j++) {
                    outputMatrix[i][j] += biasData.get(j);
                }
            }

            output = new Variable(NdArray.of(outputMatrix));
        }

        // 重塑输出为4维 (batch_size, output_channels, output_height, output_width)
        Shape outputShape = Shape.of(batchSize, outChannels, outputHeight, outputWidth);
        output = output.reshape(outputShape);

        return output;
    }

    /**
     * 执行Im2Col转换
     */
    private NdArray performIm2Col(NdArray inputData, int kernelH, int kernelW, int stride, int pad) {
        // 这里实现一个简化版本的Im2Col
        // 实际应用中应该使用Im2ColUtil类

        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);

        int outHeight = (height + 2 * pad - kernelH) / stride + 1;
        int outWidth = (width + 2 * pad - kernelW) / stride + 1;

        // 创建输出矩阵
        int outputRows = batchSize * outHeight * outWidth;
        int outputCols = channels * kernelH * kernelW;

        Shape outputShape = Shape.of(outputRows, outputCols);
        float[] outputData = new float[outputRows * outputCols];

        int outputRowIndex = 0;
        for (int n = 0; n < batchSize; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int colIndex = 0;

                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < kernelH; fh++) {
                            int imRow = h * stride + fh - pad;
                            for (int fw = 0; fw < kernelW; fw++) {
                                int imCol = w * stride + fw - pad;

                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    outputData[outputRowIndex * outputCols + colIndex] =
                                            inputData.get(n, c, imRow, imCol);
                                } else {
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

        return NdArray.of(outputData, outputShape);
    }

    /**
     * 重塑权重为二维矩阵
     */
    private NdArray reshapeWeight() {
        // 权重形状从 (out_channels, in_channels, kernel_h, kernel_w)
        // 重塑为 (out_channels, in_channels * kernel_h * kernel_w)
        NdArray weightData = weight.getValue();
        Shape newShape = Shape.of(outChannels, inChannels * kernelHeight * kernelWidth);

        return weightData.reshape(newShape);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward0(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 采用基于im2col的简化反向传播：
        // y = im2col(x) * W^T + b
        // dW = dy^T * im2col(x)
        // dx_col = dy * W
        // dx 通过 col2im 还原到输入形状

        if (lastIm2col == null || lastInput == null) {
            throw new IllegalStateException("Backward called before forward in ConvLayer");
        }

        int outSpatial = lastBatchSize * lastOutHeight * lastOutWidth;
        int colDim = inChannels * kernelHeight * kernelWidth;

        // 1) 处理上游梯度形状: (batch, out_c, out_h, out_w) -> (outSpatial, out_c)
        NdArray gradOutReshaped = yGrad.reshape(Shape.of(outSpatial, outChannels));
        float[][] gradOutMat = gradOutReshaped.getMatrix();

        float[][] im2colMat = lastIm2col.getMatrix(); // shape (outSpatial, colDim)

        // 2) 计算权重梯度 (out_c, colDim)
        float[][] weightGrad2D = new float[outChannels][colDim];
        for (int oc = 0; oc < outChannels; oc++) {
            for (int k = 0; k < colDim; k++) {
                float sum = 0f;
                for (int i = 0; i < outSpatial; i++) {
                    sum += gradOutMat[i][oc] * im2colMat[i][k];
                }
                weightGrad2D[oc][k] = sum;
            }
        }
        NdArray weightGrad = NdArray.of(weightGrad2D)
                .reshape(Shape.of(outChannels, inChannels, kernelHeight, kernelWidth));
        weight.setGrad(weightGrad);

        // 3) 计算偏置梯度 (out_c)
        if (useBias && bias != null) {
            float[] biasGrad = new float[outChannels];
            for (int i = 0; i < outSpatial; i++) {
                for (int oc = 0; oc < outChannels; oc++) {
                    biasGrad[oc] += gradOutMat[i][oc];
                }
            }
            bias.setGrad(NdArray.of(biasGrad, Shape.of(outChannels)));
        }

        // 4) 计算输入梯度（先得到列形式，再 col2im）
        float[][] weight2D = reshapeWeight().getMatrix(); // (out_c, colDim)
        float[][] gradXCol = new float[outSpatial][colDim];
        for (int i = 0; i < outSpatial; i++) {
            for (int k = 0; k < colDim; k++) {
                float sum = 0f;
                for (int oc = 0; oc < outChannels; oc++) {
                    sum += gradOutMat[i][oc] * weight2D[oc][k];
                }
                gradXCol[i][k] = sum;
            }
        }

        NdArray gradInput = col2im(gradXCol);

        List<NdArray> result = new ArrayList<>();
        result.add(gradInput);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 将 im2col 展开的梯度还原为输入形状
     */
    private NdArray col2im(float[][] gradCols) {
        int height = lastInput.getShape().getDimension(2);
        int width = lastInput.getShape().getDimension(3);
        float[] gradInputData = new float[lastBatchSize * inChannels * height * width];

        int rowIndex = 0;
        for (int n = 0; n < lastBatchSize; n++) {
            for (int h = 0; h < lastOutHeight; h++) {
                for (int w = 0; w < lastOutWidth; w++) {
                    int colIndex = 0;
                    for (int c = 0; c < inChannels; c++) {
                        for (int fh = 0; fh < kernelHeight; fh++) {
                            int imRow = h * stride + fh - padding;
                            for (int fw = 0; fw < kernelWidth; fw++) {
                                int imCol = w * stride + fw - padding;
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    int inputIdx = ((n * inChannels + c) * height + imRow) * width + imCol;
                                    gradInputData[inputIdx] += gradCols[rowIndex][colIndex];
                                }
                                colIndex++;
                            }
                        }
                    }
                    rowIndex++;
                }
            }
        }
        return NdArray.of(gradInputData, Shape.of(lastBatchSize, inChannels, height, width));
    }
}