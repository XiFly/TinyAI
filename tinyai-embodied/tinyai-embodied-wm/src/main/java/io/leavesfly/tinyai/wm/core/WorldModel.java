package io.leavesfly.tinyai.wm.core;

import io.leavesfly.tinyai.agent.wm.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.wm.model.*;

/**
 * 世界模型（World Model）
 * 整合VAE编码器、MDN-RNN和控制器的完整世界模型
 * 
 * 核心思想：
 * 1. VAE：将高维观察压缩为低维潜在表示
 * 2. MDN-RNN：在潜在空间中预测动态演化
 * 3. Controller：基于潜在表示选择动作
 * 
 * 工作流程：
 * - 训练阶段：
 *   1. 收集观察数据训练VAE
 *   2. 在潜在空间中收集序列数据训练MDN-RNN
 *   3. 在想象环境中训练Controller
 * - 推理阶段：
 *   1. VAE编码当前观察
 *   2. MDN-RNN预测未来状态
 *   3. Controller选择动作
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class WorldModel {
    
    /**
     * VAE编码器
     */
    private final VAEEncoder vaeEncoder;
    
    /**
     * MDN-RNN记忆网络
     */
    private final MDNRNN mdnRnn;
    
    /**
     * 控制器
     */
    private final Controller controller;
    
    /**
     * 当前世界模型状态
     */
    private WorldModelState currentState;
    
    /**
     * 配置参数
     */
    private final WorldModelConfig config;
    
    /**
     * 构造函数
     *
     * @param config 世界模型配置
     */
    public WorldModel(WorldModelConfig config) {
        this.config = config;
        
        // 创建VAE编码器
        this.vaeEncoder = new VAEEncoder(
            config.getObservationSize(),
            config.getLatentSize(),
            config.getVaeHiddenSize()
        );
        
        // 创建MDN-RNN
        this.mdnRnn = new MDNRNN(
            config.getLatentSize(),
            config.getActionSize(),
            config.getHiddenSize(),
            config.getNumMixtures()
        );
        
        // 创建控制器
        this.controller = new Controller(
            config.getLatentSize(),
            config.getHiddenSize(),
            config.getActionSize(),
            config.isDeterministic()
        );
        
        // 初始化状态
        this.currentState = WorldModelState.createInitial(
            config.getLatentSize(),
            config.getHiddenSize(),
            false // 使用GRU而非LSTM
        );
        
        // 子模块独立管理，不需要注册
    }
    
    /**
     * 重置世界模型状态
     */
    public void reset() {
        this.currentState = WorldModelState.createInitial(
            config.getLatentSize(),
            config.getHiddenSize(),
            false
        );
    }
    
    /**
     * 处理新观察并更新内部状态
     *
     * @param observation 新观察
     * @return 更新后的状态
     */
    public WorldModelState updateState(Observation observation) {
        // 1. 使用VAE编码观察
        LatentState latentState = vaeEncoder.encode(observation);
        
        // 2. 更新世界模型状态
        this.currentState = new WorldModelState(latentState, currentState.getHiddenState());
        
        return currentState;
    }
    
    /**
     * 选择动作
     *
     * @return 选择的动作
     */
    public Action selectAction() {
        return controller.selectAction(currentState);
    }
    
    /**
     * 完整的一步交互：观察 -> 编码 -> 选择动作 -> 预测下一状态
     *
     * @param observation 当前观察
     * @return 选择的动作
     */
    public Action step(Observation observation) {
        // 1. 更新状态（编码观察）
        updateState(observation);
        
        // 2. 选择动作
        Action action = selectAction();
        
        // 3. 预测下一个状态（用于内部规划）
        predictNextState(action);
        
        return action;
    }
    
    /**
     * 预测下一个状态（基于当前状态和动作）
     *
     * @param action 动作
     * @return 预测的下一个状态
     */
    public WorldModelState predictNextState(Action action) {
        // 使用MDN-RNN预测
        MDNRNN.RNNOutput output = mdnRnn.forward(
            currentState.getLatentState(),
            action,
            currentState.getHiddenState()
        );
        
        // 从MDN分布中采样下一个潜在状态
        LatentState nextLatent = mdnRnn.sample(output.getMdnParams());
        
        // 更新隐藏状态
        HiddenState nextHidden = output.getNextHidden();
        
        // 更新当前状态
        this.currentState = new WorldModelState(nextLatent, nextHidden);
        
        return currentState;
    }
    
    /**
     * 在想象环境中rollout（不与真实环境交互）
     * 用于在潜在空间中进行规划和训练
     *
     * @param initialState 初始状态
     * @param steps rollout步数
     * @return 想象的情景
     */
    public Episode dreamRollout(WorldModelState initialState, int steps) {
        Episode dreamEpisode = new Episode("dream_" + System.currentTimeMillis());
        WorldModelState state = initialState.copy();
        
        for (int t = 0; t < steps; t++) {
            // 1. 选择动作
            Action action = controller.selectAction(state);
            
            // 2. 预测下一状态
            MDNRNN.RNNOutput output = mdnRnn.forward(
                state.getLatentState(),
                action,
                state.getHiddenState()
            );
            LatentState nextLatent = mdnRnn.sample(output.getMdnParams());
            HiddenState nextHidden = output.getNextHidden();
            WorldModelState nextState = new WorldModelState(nextLatent, nextHidden);
            
            // 3. 计算想象的奖励（基于潜在状态的某种启发式）
            double reward = calculateImaginedReward(state, action, nextState);
            
            // 4. 创建虚拟观察（从潜在状态解码）
            Observation obs = createDummyObservation(state.getLatentState());
            Observation nextObs = createDummyObservation(nextLatent);
            
            // 5. 记录转换
            Transition transition = new Transition(obs, action, reward, nextObs, false, t);
            dreamEpisode.addTransition(transition);
            
            state = nextState;
        }
        
        return dreamEpisode;
    }
    
    /**
     * 计算想象的奖励（启发式方法）
     */
    private double calculateImaginedReward(WorldModelState state, Action action, WorldModelState nextState) {
        // 简化实现：奖励与潜在状态的范数成反比（鼓励探索）
        NdArray z = state.getLatentState().getZ();
        double stateNorm = Math.sqrt(z.mul(z).sum().getNumber().doubleValue());
        NdArray actionVec = action.getActionVector();
        double actionCost = actionVec.mul(actionVec).sum().getNumber().doubleValue() / actionVec.getShape().size() * 0.01;
        return -stateNorm * 0.1 - actionCost;
    }
    
    /**
     * 创建虚拟观察（用于想象rollout）
     */
    private Observation createDummyObservation(LatentState latentState) {
        // 从潜在状态解码
        NdArray reconstructed = vaeEncoder.decode(latentState);
        NdArray dummyVisual = NdArray.zeros(Shape.of(3, 64, 64));
        return new Observation(dummyVisual, reconstructed);
    }
    
    // 不继承Model，移除forward方法
    
    // Getters
    public VAEEncoder getVaeEncoder() {
        return vaeEncoder;
    }
    
    public MDNRNN getMdnRnn() {
        return mdnRnn;
    }
    
    public Controller getController() {
        return controller;
    }
    
    public WorldModelState getCurrentState() {
        return currentState;
    }
    
    public WorldModelConfig getConfig() {
        return config;
    }
    
    /**
     * 世界模型配置类
     */
    public static class WorldModelConfig {
        private final int observationSize;
        private final int latentSize;
        private final int hiddenSize;
        private final int actionSize;
        private final int vaeHiddenSize;
        private final int numMixtures;
        private final boolean deterministic;
        
        public WorldModelConfig(int observationSize, int latentSize, int hiddenSize,
                               int actionSize, int vaeHiddenSize, int numMixtures,
                               boolean deterministic) {
            this.observationSize = observationSize;
            this.latentSize = latentSize;
            this.hiddenSize = hiddenSize;
            this.actionSize = actionSize;
            this.vaeHiddenSize = vaeHiddenSize;
            this.numMixtures = numMixtures;
            this.deterministic = deterministic;
        }
        
        /**
         * 创建默认配置
         */
        public static WorldModelConfig createDefault() {
            return new WorldModelConfig(
                8,     // observationSize (与 SimpleDrivingEnvironment 一致)
                32,    // latentSize
                256,   // hiddenSize
                3,     // actionSize
                128,   // vaeHiddenSize
                5,     // numMixtures
                false  // deterministic
            );
        }
        
        // Getters
        public int getObservationSize() { return observationSize; }
        public int getLatentSize() { return latentSize; }
        public int getHiddenSize() { return hiddenSize; }
        public int getActionSize() { return actionSize; }
        public int getVaeHiddenSize() { return vaeHiddenSize; }
        public int getNumMixtures() { return numMixtures; }
        public boolean isDeterministic() { return deterministic; }
    }
}
