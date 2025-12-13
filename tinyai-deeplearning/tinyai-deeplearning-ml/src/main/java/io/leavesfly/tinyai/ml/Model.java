package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.inference.Predictor;
import io.leavesfly.tinyai.ml.inference.Translator;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Block;
import io.leavesfly.tinyai.nnet.v1.ParameterV1; // V1 ParameterV1 (for backward compatibility)
import io.leavesfly.tinyai.nnet.V2ParameterProxy; // V2 Parameter 的代理类
import io.leavesfly.tinyai.nnet.v2.adapter.BlockModuleAdapter;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.*;
import java.util.Map;

/**
 * 机器学习模型类
 * <p>
 * 该类是TinyDL框架中模型的核心表示，提供了模型的完整生命周期管理功能，
 * 包括模型的创建、训练、保存、加载、推理等操作。
 * <p>
 * 主要功能：
 * 1. 模型结构管理：封装神经网络的架构（V2 Module，兼容 V1 Block）
 * 2. 模型序列化：支持完整模型、参数、检查点等多种保存方式
 * 3. 模型推理：提供预测接口
 * 4. 模型信息管理：维护模型的元数据信息
 * <p>
 * 版本说明：
 * - V2.0: 默认支持 V2 Module，通过适配器兼容 V1 Block
 * - V1.0: 仅支持 V1 Block（已废弃，但保持向后兼容）
 *
 * @author TinyDL
 * @version 2.0
 */
public class Model implements Serializable {

    private static final long serialVersionUID = 2L;

    private String name;

    /**
     * 模型的核心结构（V2 Module）
     * 如果传入的是 V1 Block，会自动适配为 Module
     */
    private Module module;

    /**
     * 如果是通过 V1 Block 适配的，保存原始 Block 引用（用于序列化兼容）
     */
    private transient Block originalBlock;

    // 模型元数据信息
    private ModelInfo modelInfo;

    public transient Variable tmpPredict;

    /**
     * 构造函数 - 使用 V2 Module（推荐）
     *
     * @param _name   模型名称
     * @param _module 模型的神经网络结构（V2 Module）
     */
    public Model(String _name, Module _module) {
        name = _name;
        module = _module;
        modelInfo = new ModelInfo(_name);
        initializeModelInfo();
    }

    /**
     * 构造函数 - 使用 V1 Block（向后兼容）
     * 内部会自动将 Block 适配为 Module
     *
     * @param _name  模型名称
     * @param _block 模型的神经网络结构（V1 Block）
     */
    public Model(String _name, Block _block) {
        name = _name;
        originalBlock = _block;
        // 自动适配 V1 Block 为 V2 Module
        module = new BlockModuleAdapter(_block);
        modelInfo = new ModelInfo(_name);
        initializeModelInfo();
    }

    /**
     * 初始化模型信息
     * 包括输入输出形状、参数数量、架构类型等基本信息
     */
    private void initializeModelInfo() {
        if (module != null) {
            // 获取输入输出形状（Module 可能没有，通过适配器或延迟推断）
            Shape inputShape = getInputShape();
            Shape outputShape = getOutputShape();
            modelInfo.setInputShape(inputShape);
            modelInfo.setOutputShape(outputShape);

            // 统计参数数量（使用 V2 接口）
            Map<String,Parameter> params = module.namedParameters();
            long totalParams = 0;
            for (Parameter param : params.values()) {
                if (param != null && param.data() != null) {
                    totalParams += param.data().getShape().size();
                }
            }
            modelInfo.setTotalParameters(totalParams);

            // 设置架构类型（根据 Module 类型推断）
            String moduleClassName = module.getClass().getSimpleName();
            // 如果是适配器，使用原始 Block 的类型名
            if (module instanceof BlockModuleAdapter) {
                BlockModuleAdapter adapter = (BlockModuleAdapter) module;
                Block block = adapter.getBlock();
                if (block != null) {
                    moduleClassName = block.getClass().getSimpleName();
                }
            }
            modelInfo.setArchitectureType(moduleClassName);
        }
    }

    /**
     * 获取输入形状
     * 如果 Module 是适配器，尝试从适配器获取；否则返回 null
     *
     * @return 输入形状，如果无法获取则返回 null
     */
    private Shape getInputShape() {
        if (module instanceof BlockModuleAdapter) {
            BlockModuleAdapter adapter = (BlockModuleAdapter) module;
            return adapter.getInputShape();
        }
        // V2 Module 没有 getInputShape，返回 null
        return null;
    }

    /**
     * 获取输出形状
     * 如果 Module 是适配器，尝试从适配器获取；否则返回 null
     *
     * @return 输出形状，如果无法获取则返回 null
     */
    private Shape getOutputShape() {
        if (module instanceof BlockModuleAdapter) {
            BlockModuleAdapter adapter = (BlockModuleAdapter) module;
            return adapter.getOutputShape();
        }
        // V2 Module 没有 getOutputShape，返回 null
        return null;
    }

    /**
     * 绘制模型计算图
     * 通过可视化方式展示模型的前向传播计算过程
     */
    public void plot() {
        Shape inputShape = getInputShape();
        if (inputShape != null) {
            tmpPredict = module.forward(new Variable(NdArray.ones(inputShape)));
        }
        // System.out.println(Uml.getDotGraph(tmpPredict)); // 暂时注释，因为Uml类不可用
    }

    /**
     * 保存模型到文件（传统方式）
     *
     * @param modelFile 模型文件
     */
    public void save(File modelFile) {
        try (FileOutputStream fileOut = new FileOutputStream(modelFile);
             ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(this);
        } catch (Exception e) {
            throw new RuntimeException("Model save error!");
        }
    }

    /**
     * 使用ModelSerializer保存模型（推荐使用）
     *
     * @param filePath 保存路径
     */
    public void saveModel(String filePath) {
        ModelSerializer.saveModel(this, filePath);
    }

    /**
     * 保存压缩模型
     *
     * @param filePath 保存路径
     */
    public void saveModelCompressed(String filePath) {
        ModelSerializer.saveModel(this, filePath, true);
    }

    /**
     * 仅保存模型参数
     *
     * @param filePath 保存路径
     */
    public void saveParameters(String filePath) {
        ModelSerializer.saveParameters(this, filePath);
    }

    /**
     * 保存训练检查点
     *
     * @param filePath 保存路径
     * @param epoch    当前训练轮次
     * @param loss     当前损失
     */
    public void saveCheckpoint(String filePath, int epoch, double loss) {
        // 更新模型信息
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epoch);
            modelInfo.setFinalLoss(loss);
            if (loss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(loss);
            }
        }
        ModelSerializer.saveCheckpoint(this, epoch, loss, filePath);
    }

    /**
     * 从文件加载模型（传统方式）
     *
     * @param modelFile 模型文件
     * @return 加载的模型
     */
    public static Model load(File modelFile) {
        try (FileInputStream fileIn = new FileInputStream(modelFile);
             ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Model) in.readObject();
        } catch (Exception e) {
            throw new RuntimeException("model load error!");
        }
    }

    /**
     * 使用ModelSerializer加载模型（推荐使用）
     *
     * @param filePath 模型文件路径
     * @return 加载的模型
     */
    public static Model loadModel(String filePath) {
        return ModelSerializer.loadModel(filePath);
    }

    /**
     * 加载参数到当前模型
     *
     * @param filePath 参数文件路径
     */
    public void loadParameters(String filePath) {
        ModelSerializer.loadParameters(this, filePath);
    }

    /**
     * 从检查点恢复模型
     *
     * @param filePath 检查点文件路径
     * @return 恢复的模型
     */
    public static Model resumeFromCheckpoint(String filePath) {
        return ModelSerializer.resumeFromCheckpoint(filePath);
    }

    /**
     * 重置模型状态
     * 主要用于RNN等有状态的模型，清除历史状态信息
     * 如果 Module 是适配器，调用适配器的 resetState 方法
     */
    public void resetState() {
        if (module instanceof BlockModuleAdapter) {
            BlockModuleAdapter adapter = (BlockModuleAdapter) module;
            adapter.resetState();
        }
        // V2 Module 没有 resetState 方法，如果需要可以扩展
    }

    /**
     * 模型前向传播
     *
     * @param inputs 输入变量
     * @return 输出变量
     */
    public Variable forward(Variable... inputs) {
        return module.forward(inputs);
    }

    /**
     * 清除梯度
     * 在每次反向传播前调用，清除历史梯度信息
     */
    public void clearGrads() {
        module.clearGrads();
    }

    /**
     * 获取所有参数（V1 兼容接口）
     * 返回 V1 格式的参数映射，用于向后兼容
     *
     * @return 参数映射（V1 ParameterV1 格式）
     */
    public Map<String, ParameterV1> getAllParams() {
        // 如果是通过 BlockModuleAdapter 适配的 V1 Block，直接返回 Block 内部的原始参数
        // 这确保 SGD 等优化器更新的是实际使用的参数，而不是临时创建的副本
        if (module instanceof BlockModuleAdapter) {
            BlockModuleAdapter adapter = (BlockModuleAdapter) module;
            Block block = adapter.getBlock();
            if (block != null) {
                return block.getAllParams();
            }
        }
        
        // 对于纯 V2 Module，使用 V2ParameterProxy 包装 V2 Parameter
        // V2ParameterProxy 会在 setValue 时同步更新原始的 V2 Parameter
        Map<String, io.leavesfly.tinyai.nnet.v2.core.Parameter> v2Params = module.namedParameters();
        Map<String, ParameterV1> v1Params = new java.util.HashMap<>();
        
        for (Map.Entry<String, io.leavesfly.tinyai.nnet.v2.core.Parameter> entry : v2Params.entrySet()) {
            io.leavesfly.tinyai.nnet.v2.core.Parameter v2Param = entry.getValue();
            if (v2Param != null) {
                // 使用代理类，确保参数更新能同步回原始 V2 Parameter
                ParameterV1 v1Param = new V2ParameterProxy(v2Param);
                v1Params.put(entry.getKey(), v1Param);
            }
        }
        
        return v1Params;
    }

    /**
     * 获取所有参数（V2 接口）
     * 返回 V2 格式的参数映射
     *
     * @return 参数映射（V2 ParameterV1 格式）
     */
    public Map<String, Parameter> getAllParamsV2() {
        return module.namedParameters();
    }

    /**
     * 获取预测器
     *
     * @param translator 翻译器
     * @return 预测器
     */
    public <I, O> Predictor<I, O> getPredictor(Translator<I, O> translator) {
        return new Predictor<>(translator, this);
    }

    /**
     * 获取模型名称
     *
     * @return 模型名称
     */
    public String getName() {
        return name;
    }

    /**
     * 获取模型结构（V2 Module）
     *
     * @return 模型结构（V2 Module）
     */
    public Module getModule() {
        return module;
    }

    /**
     * 获取模型结构（V1 Block，向后兼容）
     * 如果 Module 是适配器，返回原始 Block；否则返回 null
     *
     * @return 模型结构块（V1 Block），如果不存在则返回 null
     * @deprecated 使用 getModule() 代替，此方法仅用于向后兼容
     */
    @Deprecated
    public Block getBlock() {
        if (module instanceof BlockModuleAdapter) {
            BlockModuleAdapter adapter = (BlockModuleAdapter) module;
            return adapter.getBlock();
        }
        return originalBlock;
    }

    // =========== 模型信息相关方法 ===========

    /**
     * 获取模型信息
     *
     * @return 模型信息
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * 设置模型信息
     *
     * @param modelInfo 模型信息
     */
    public void setModelInfo(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    /**
     * 更新训练信息
     *
     * @param epochs       训练轮次
     * @param finalLoss    最终损失
     * @param optimizer    优化器名称
     * @param learningRate 学习率
     */
    public void updateTrainingInfo(int epochs, double finalLoss, String optimizer, double learningRate) {
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epochs);
            modelInfo.setFinalLoss(finalLoss);
            modelInfo.setOptimizerType(optimizer);
            modelInfo.setLearningRate(learningRate);

            if (finalLoss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(finalLoss);
            }
        }
    }

    /**
     * 添加性能指标
     *
     * @param metricName 指标名称
     * @param value      指标值
     */
    public void addMetric(String metricName, double value) {
        if (modelInfo != null) {
            modelInfo.addMetric(metricName, value);
        }
    }

    /**
     * 设置模型描述
     *
     * @param description 描述
     */
    public void setDescription(String description) {
        if (modelInfo != null) {
            modelInfo.setDescription(description);
        }
    }

    /**
     * 获取模型简要信息
     *
     * @return 简要信息
     */
    public String getModelSummary() {
        if (modelInfo != null) {
            return modelInfo.getSummary();
        }
        return "Model: " + name + "\nNo detailed info available.";
    }

    /**
     * 获取模型详细信息
     *
     * @return 详细信息
     */
    public String getModelDetailedInfo() {
        if (modelInfo != null) {
            return modelInfo.getDetailedInfo();
        }
        return getModelSummary();
    }

    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println(getModelDetailedInfo());
    }

    /**
     * 验证模型文件
     *
     * @param filePath 文件路径
     * @return 是否有效
     */
    public static boolean validateModel(String filePath) {
        return ModelSerializer.validateModelFile(filePath);
    }

}