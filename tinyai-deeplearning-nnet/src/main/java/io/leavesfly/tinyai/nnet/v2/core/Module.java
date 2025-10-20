package io.leavesfly.tinyai.nnet.v2.core;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;
import java.util.function.Consumer;

/**
 * V2版本的神经网络模块基类
 * <p>
 * Module继承自Function，保持自动微分能力，同时提供增强的参数管理功能。
 * 核心特性：
 * - 统一的参数和缓冲区注册机制
 * - 自动的分层命名路径管理
 * - 训练/推理模式切换
 * - 状态字典序列化支持
 *
 * @author leavesfly
 * @version 2.0
 */
public abstract class Module extends Function {

    /**
     * 模块名称
     */
    protected String name;

    /**
     * 可训练参数集合
     * key: 参数名称, value: Parameter对象
     */
    protected Map<String, Parameter> _parameters;

    /**
     * 非可训练缓冲区集合（如BatchNorm的running_mean）
     * key: 缓冲区名称, value: NdArray对象
     */
    protected Map<String, NdArray> _buffers;

    /**
     * 子模块集合
     * key: 子模块名称, value: Module对象
     */
    protected Map<String, Module> _modules;

    /**
     * 父模块引用
     */
    protected Module _parent;

    /**
     * 是否处于训练模式
     */
    protected boolean _training = true;

    /**
     * 是否已初始化
     */
    protected boolean _initialized = false;

    /**
     * 默认构造函数
     */
    public Module() {
        this(null);
    }

    /**
     * 带名称的构造函数
     *
     * @param name 模块名称
     */
    public Module(String name) {
        this.name = name;
        this._parameters = new LinkedHashMap<>();
        this._buffers = new LinkedHashMap<>();
        this._modules = new LinkedHashMap<>();
    }

    /**
     * 注册可训练参数
     * <p>
     * 统一的参数注册接口，确保参数被正确管理。
     * 会检查名称冲突，参数可以为null（延迟初始化场景）。
     *
     * @param paramName 参数名称
     * @param param     参数对象（可为null）
     * @return 注册的参数对象
     * @throws IllegalArgumentException 如果名称与已有参数、缓冲区或子模块冲突
     */
    public Parameter registerParameter(String paramName, Parameter param) {
        if (_parameters.containsKey(paramName) ||
                _buffers.containsKey(paramName) ||
                _modules.containsKey(paramName)) {
            throw new IllegalArgumentException(
                    "Parameter name '" + paramName + "' conflicts with existing parameter, buffer or module");
        }
        _parameters.put(paramName, param);
        return param;
    }

    /**
     * 注册非可训练缓冲区
     * <p>
     * 用于管理不参与梯度计算但需要序列化的张量，
     * 如BatchNorm的running_mean、running_var等。
     *
     * @param bufferName 缓冲区名称
     * @param buffer     缓冲区张量（可为null）
     * @throws IllegalArgumentException 如果名称与已有参数、缓冲区或子模块冲突
     */
    public void registerBuffer(String bufferName, NdArray buffer) {
        if (_parameters.containsKey(bufferName) ||
                _buffers.containsKey(bufferName) ||
                _modules.containsKey(bufferName)) {
            throw new IllegalArgumentException(
                    "Buffer name '" + bufferName + "' conflicts with existing parameter, buffer or module");
        }
        _buffers.put(bufferName, buffer);
    }

    /**
     * 注册子模块
     * <p>
     * 自动设置子模块的父引用，子模块的参数自动纳入父模块管理。
     *
     * @param moduleName 子模块名称
     * @param module     子模块对象
     * @return 注册的子模块对象
     * @throws IllegalArgumentException 如果名称与已有参数、缓冲区或子模块冲突
     */
    public Module registerModule(String moduleName, Module module) {
        if (_parameters.containsKey(moduleName) ||
                _buffers.containsKey(moduleName) ||
                _modules.containsKey(moduleName)) {
            throw new IllegalArgumentException(
                    "Module name '" + moduleName + "' conflicts with existing parameter, buffer or module");
        }
        if (module != null) {
            module._parent = this;
        }
        _modules.put(moduleName, module);
        return module;
    }

    /**
     * 获取指定名称的参数
     *
     * @param paramName 参数名称
     * @return 参数对象，如果不存在返回null
     */
    public Parameter getParameter(String paramName) {
        return _parameters.get(paramName);
    }

    /**
     * 获取指定名称的缓冲区
     *
     * @param bufferName 缓冲区名称
     * @return 缓冲区张量，如果不存在返回null
     */
    public NdArray getBuffer(String bufferName) {
        return _buffers.get(bufferName);
    }

    /**
     * 获取指定名称的子模块
     *
     * @param moduleName 子模块名称
     * @return 子模块对象，如果不存在返回null
     */
    public Module getModule(String moduleName) {
        return _modules.get(moduleName);
    }

    /**
     * 递归收集所有参数及其完整路径
     * <p>
     * 自动构建分层命名路径，如 "encoder.layer1.weight"
     *
     * @param prefix  路径前缀
     * @param recurse 是否递归遍历子模块
     * @return 完整路径到参数的映射
     */
    public Map<String, Parameter> namedParameters(String prefix, boolean recurse) {
        Map<String, Parameter> result = new LinkedHashMap<>();

        // 收集当前模块的参数
        for (Map.Entry<String, Parameter> entry : _parameters.entrySet()) {
            String fullName = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            if (entry.getValue() != null) {
                result.put(fullName, entry.getValue());
            }
        }

        // 递归收集子模块的参数
        if (recurse) {
            for (Map.Entry<String, Module> entry : _modules.entrySet()) {
                if (entry.getValue() != null) {
                    String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
                    result.putAll(entry.getValue().namedParameters(childPrefix, true));
                }
            }
        }

        return result;
    }

    /**
     * 收集所有参数（默认递归）
     *
     * @return 完整路径到参数的映射
     */
    public Map<String, Parameter> namedParameters() {
        return namedParameters("", true);
    }

    /**
     * 递归收集所有缓冲区及其完整路径
     *
     * @param prefix  路径前缀
     * @param recurse 是否递归遍历子模块
     * @return 完整路径到缓冲区的映射
     */
    public Map<String, NdArray> namedBuffers(String prefix, boolean recurse) {
        Map<String, NdArray> result = new LinkedHashMap<>();

        // 收集当前模块的缓冲区
        for (Map.Entry<String, NdArray> entry : _buffers.entrySet()) {
            String fullName = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            if (entry.getValue() != null) {
                result.put(fullName, entry.getValue());
            }
        }

        // 递归收集子模块的缓冲区
        if (recurse) {
            for (Map.Entry<String, Module> entry : _modules.entrySet()) {
                if (entry.getValue() != null) {
                    String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
                    result.putAll(entry.getValue().namedBuffers(childPrefix, true));
                }
            }
        }

        return result;
    }

    /**
     * 收集所有缓冲区（默认递归）
     *
     * @return 完整路径到缓冲区的映射
     */
    public Map<String, NdArray> namedBuffers() {
        return namedBuffers("", true);
    }

    /**
     * 递归收集所有子模块及其完整路径
     *
     * @param prefix  路径前缀
     * @param recurse 是否递归遍历子模块
     * @return 完整路径到子模块的映射
     */
    public Map<String, Module> namedModules(String prefix, boolean recurse) {
        Map<String, Module> result = new LinkedHashMap<>();

        // 添加当前模块
        if (!prefix.isEmpty()) {
            result.put(prefix, this);
        }

        // 遍历子模块
        for (Map.Entry<String, Module> entry : _modules.entrySet()) {
            if (entry.getValue() != null) {
                String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
                if (recurse) {
                    result.putAll(entry.getValue().namedModules(childPrefix, true));
                } else {
                    result.put(childPrefix, entry.getValue());
                }
            }
        }

        return result;
    }

    /**
     * 收集所有子模块（默认递归）
     *
     * @return 完整路径到子模块的映射
     */
    public Map<String, Module> namedModules() {
        return namedModules("", true);
    }

    /**
     * 设置训练/推理模式
     * <p>
     * 递归设置当前模块及所有子模块的模式
     *
     * @param mode true为训练模式，false为推理模式
     * @return 当前模块（支持链式调用）
     */
    public Module train(boolean mode) {
        _training = mode;
        for (Module child : _modules.values()) {
            if (child != null) {
                child.train(mode);
            }
        }
        return this;
    }

    /**
     * 设置为训练模式
     *
     * @return 当前模块（支持链式调用）
     */
    public Module train() {
        return train(true);
    }

    /**
     * 设置为推理模式
     *
     * @return 当前模块（支持链式调用）
     */
    public Module eval() {
        return train(false);
    }

    /**
     * 判断是否处于训练模式
     *
     * @return true表示训练模式，false表示推理模式
     */
    public boolean isTraining() {
        return _training;
    }

    /**
     * 参数初始化方法
     * <p>
     * 子类重写此方法实现参数的初始化策略。
     * 应将参数创建和初始化分离，在此方法中仅进行初始化。
     */
    public void resetParameters() {
        // 默认空实现，子类按需重写
    }

    /**
     * 模块初始化方法
     * <p>
     * 在首次使用前调用，用于延迟初始化场景。
     * 标准模块在构造函数中完成参数创建，此方法调用resetParameters初始化参数值。
     */
    public void init() {
        if (!_initialized) {
            resetParameters();
            _initialized = true;
        }
    }

    /**
     * 层的前向传播
     * <p>
     * 子类必须实现此方法定义具体的前向传播逻辑。
     * 操作Variable对象以保持计算图完整性。
     *
     * @param inputs 输入变量数组
     * @return 前向传播结果
     */
    public abstract Variable forward(Variable... inputs);

    /**
     * Function接口实现：NdArray形式的前向传播
     * <p>
     * 将NdArray转换为Variable后调用forward(Variable...)
     *
     * @param inputs 输入的NdArray数组
     * @return 前向传播结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable[] varInputs = Arrays.stream(inputs)
                .map(Variable::new)
                .toArray(Variable[]::new);
        return forward(varInputs).getValue();
    }

    /**
     * Function接口实现：反向传播
     * <p>
     * 通常不直接实现，依赖Variable的自动微分机制
     *
     * @param yGrad 输出梯度
     * @return 输入梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 依赖Variable的自动微分，返回空列表
        return Collections.emptyList();
    }

    /**
     * Function接口实现：输入参数数量要求
     *
     * @return -1表示接受任意数量输入
     */
    @Override
    public int requireInputNum() {
        return -1;
    }

    /**
     * 导出模型状态字典
     * <p>
     * 包含所有参数和缓冲区的数据（不含梯度）
     *
     * @return 完整路径到张量数据的映射
     */
    public Map<String, NdArray> stateDict() {
        Map<String, NdArray> state = new LinkedHashMap<>();

        // 收集参数
        for (Map.Entry<String, Parameter> entry : namedParameters().entrySet()) {
            if (entry.getValue() != null && entry.getValue().getValue() != null) {
                NdArray data = entry.getValue().getValue();
                // 创建数据副本
                NdArray dataCopy = data.getShape().isMatrix() ?
                        NdArray.of(data.getMatrix()) : NdArray.of(data.getArray(), data.getShape());
                state.put(entry.getKey(), dataCopy);
            }
        }

        // 收集缓冲区
        for (Map.Entry<String, NdArray> entry : namedBuffers().entrySet()) {
            if (entry.getValue() != null) {
                NdArray buffer = entry.getValue();
                // 创建缓冲区副本
                NdArray bufferCopy = buffer.getShape().isMatrix() ?
                        NdArray.of(buffer.getMatrix()) : NdArray.of(buffer.getArray(), buffer.getShape());
                state.put(entry.getKey(), bufferCopy);
            }
        }

        return state;
    }

    /**
     * 加载模型状态字典
     * <p>
     * 从状态字典恢复参数和缓冲区的值
     *
     * @param stateDict 状态字典
     * @param strict    是否严格匹配（键必须完全一致）
     * @throws IllegalArgumentException 严格模式下键不匹配时抛出异常
     */
    public void loadStateDict(Map<String, NdArray> stateDict, boolean strict) {
        Map<String, Parameter> params = namedParameters();
        Map<String, NdArray> buffers = namedBuffers();

        Set<String> loadedKeys = new HashSet<>();

        // 加载参数
        for (Map.Entry<String, NdArray> entry : stateDict.entrySet()) {
            String key = entry.getKey();
            NdArray value = entry.getValue();

            if (params.containsKey(key)) {
                Parameter param = params.get(key);
                if (param != null) {
                    // 创建值的副本
                    NdArray valueCopy = value.getShape().isMatrix() ?
                            NdArray.of(value.getMatrix()) : NdArray.of(value.getArray(), value.getShape());
                    param.setValue(valueCopy);
                    loadedKeys.add(key);
                }
            } else if (buffers.containsKey(key)) {
                // 创建缓冲区副本
                NdArray valueCopy = value.getShape().isMatrix() ?
                        NdArray.of(value.getMatrix()) : NdArray.of(value.getArray(), value.getShape());
                _buffers.put(key.substring(key.lastIndexOf('.') + 1), valueCopy);
                loadedKeys.add(key);
            } else if (strict) {
                throw new IllegalArgumentException("Unexpected key in state dict: " + key);
            }
        }

        // 严格模式下检查缺失的键
        if (strict) {
            Set<String> expectedKeys = new HashSet<>();
            expectedKeys.addAll(params.keySet());
            expectedKeys.addAll(buffers.keySet());
            expectedKeys.removeAll(loadedKeys);
            if (!expectedKeys.isEmpty()) {
                throw new IllegalArgumentException("Missing keys in state dict: " + expectedKeys);
            }
        }
    }

    /**
     * 加载模型状态字典（默认严格模式）
     *
     * @param stateDict 状态字典
     */
    public void loadStateDict(Map<String, NdArray> stateDict) {
        loadStateDict(stateDict, true);
    }

    /**
     * 对当前模块及所有子模块应用函数
     * <p>
     * 用于批量操作，如统一初始化、冻结参数等
     *
     * @param fn 应用到每个模块的函数
     */
    public void apply(Consumer<Module> fn) {
        fn.accept(this);
        for (Module child : _modules.values()) {
            if (child != null) {
                child.apply(fn);
            }
        }
    }

    /**
     * 清除所有参数的梯度
     */
    public void clearGrads() {
        for (Parameter param : _parameters.values()) {
            if (param != null) {
                param.clearGrad();
            }
        }
        for (Module child : _modules.values()) {
            if (child != null) {
                child.clearGrads();
            }
        }
    }

    /**
     * 获取模块名称
     *
     * @return 模块名称
     */
    public String getName() {
        return name;
    }

    /**
     * 设置模块名称
     *
     * @param name 模块名称
     */
    public void setName(String name) {
        this.name = name;
    }


    /**
     * 获取父模块
     *
     * @return 父模块引用
     */
    public Module getParent() {
        return _parent;
    }

    public void setParent(Module _parent) {
        this._parent = _parent;
    }

}
