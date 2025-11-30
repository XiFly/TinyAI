package io.leavesfly.tinyai.wm;

import io.leavesfly.tinyai.wm.core.WorldModel;
import io.leavesfly.tinyai.wm.env.Environment;

import io.leavesfly.tinyai.wm.model.*;

/**
 * 世界模型智能体
 * 整合世界模型与环境交互的完整智能体系统
 * 
 * 工作流程：
 * 1. 从环境获取观察
 * 2. 使用世界模型处理观察并选择动作
 * 3. 执行动作并获得奖励
 * 4. 收集经验用于训练
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class WorldModelAgent {
    
    /**
     * 世界模型
     */
    private final WorldModel worldModel;
    
    /**
     * 环境
     */
    private final Environment environment;
    
    /**
     * 当前情景
     */
    private Episode currentEpisode;
    
    /**
     * 步数计数器
     */
    private int totalSteps;
    
    /**
     * 情景计数器
     */
    private int episodeCount;
    
    /**
     * 是否处于训练模式
     */
    private boolean trainingMode;
    
    /**
     * 构造函数
     *
     * @param worldModel 世界模型
     * @param environment 环境
     */
    public WorldModelAgent(WorldModel worldModel, Environment environment) {
        this.worldModel = worldModel;
        this.environment = environment;
        this.totalSteps = 0;
        this.episodeCount = 0;
        this.trainingMode = true;
    }
    
    /**
     * 重置智能体和环境
     *
     * @return 初始观察
     */
    public Observation reset() {
        // 重置世界模型
        worldModel.reset();
        
        // 重置环境
        Observation initialObs = environment.reset();
        
        // 创建新情景
        episodeCount++;
        currentEpisode = new Episode("episode_" + episodeCount);
        
        // 更新世界模型状态
        worldModel.updateState(initialObs);
        
        return initialObs;
    }
    
    /**
     * 执行一步交互
     *
     * @return 步进结果
     */
    public Environment.StepResult step() {
        // 1. 使用世界模型选择动作
        Action action = worldModel.selectAction();
        
        // 2. 在环境中执行动作
        Environment.StepResult result = environment.step(action);
        
        // 3. 记录转换
        if (currentEpisode != null) {
            Observation currentObs = worldModel.getCurrentState().getLatentState() != null
                ? createObservationFromState()
                : result.getObservation();
            
            Transition transition = new Transition(
                currentObs,
                action,
                result.getReward(),
                result.getObservation(),
                result.isDone(),
                totalSteps
            );
            currentEpisode.addTransition(transition);
        }
        
        // 4. 更新世界模型状态
        worldModel.updateState(result.getObservation());
        
        // 5. 更新计数器
        totalSteps++;
        
        return result;
    }
    
    /**
     * 运行一个完整情景
     *
     * @param maxSteps 最大步数
     * @return 完成的情景
     */
    public Episode runEpisode(int maxSteps) {
        reset();
        
        for (int step = 0; step < maxSteps; step++) {
            Environment.StepResult result = step();
            
            if (result.isDone()) {
                System.out.println("情景结束: " + result.getInfo() + 
                                 ", 步数: " + step + 
                                 ", 总奖励: " + currentEpisode.getTotalReward());
                break;
            }
        }
        
        return currentEpisode;
    }
    
    /**
     * 在想象环境中训练（使用世界模型内部模拟）
     *
     * @param dreamSteps 想象步数
     * @return 想象的情景
     */
    public Episode trainInDream(int dreamSteps) {
        // 从当前状态开始在想象环境中rollout
        WorldModelState currentState = worldModel.getCurrentState();
        return worldModel.dreamRollout(currentState, dreamSteps);
    }
    
    /**
     * 评估智能体性能
     *
     * @param numEpisodes 评估情景数
     * @return 平均奖励
     */
    public double evaluate(int numEpisodes) {
        boolean originalMode = trainingMode;
        trainingMode = false; // 切换到评估模式
        
        double totalReward = 0.0;
        
        for (int i = 0; i < numEpisodes; i++) {
            Episode episode = runEpisode(1000);
            totalReward += episode.getTotalReward();
        }
        
        trainingMode = originalMode;
        return totalReward / numEpisodes;
    }
    
    /**
     * 从当前状态创建观察（用于记录）
     */
    private Observation createObservationFromState() {
        WorldModelState state = worldModel.getCurrentState();
        io.leavesfly.tinyai.ndarr.NdArray reconstructed = 
            worldModel.getVaeEncoder().decode(state.getLatentState());
        io.leavesfly.tinyai.ndarr.NdArray dummyVisual = 
            io.leavesfly.tinyai.ndarr.NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(3, 64, 64));
        return new Observation(dummyVisual, reconstructed);
    }
    
    /**
     * 获取统计信息
     */
    public String getStatistics() {
        return String.format(
            "智能体统计:\n" +
            "  总步数: %d\n" +
            "  情景数: %d\n" +
            "  训练模式: %b\n" +
            "  当前情景奖励: %.3f\n",
            totalSteps,
            episodeCount,
            trainingMode,
            currentEpisode != null ? currentEpisode.getTotalReward() : 0.0
        );
    }
    
    // Getters and Setters
    public WorldModel getWorldModel() {
        return worldModel;
    }
    
    public Environment getEnvironment() {
        return environment;
    }
    
    public Episode getCurrentEpisode() {
        return currentEpisode;
    }
    
    public int getTotalSteps() {
        return totalSteps;
    }
    
    public int getEpisodeCount() {
        return episodeCount;
    }
    
    public boolean isTrainingMode() {
        return trainingMode;
    }
    
    public void setTrainingMode(boolean trainingMode) {
        this.trainingMode = trainingMode;
    }
}
