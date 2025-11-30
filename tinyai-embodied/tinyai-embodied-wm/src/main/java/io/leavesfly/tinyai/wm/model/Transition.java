package io.leavesfly.tinyai.wm.model;

/**
 * 转换（Transition）
 * 表示一个时间步的完整状态转换：(s_t, a_t, r_t, s_{t+1})
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class Transition {
    
    /**
     * 当前观察
     */
    private final Observation observation;
    
    /**
     * 执行的动作
     */
    private final Action action;
    
    /**
     * 获得的奖励
     */
    private final double reward;
    
    /**
     * 下一个观察
     */
    private final Observation nextObservation;
    
    /**
     * 是否终止
     */
    private final boolean done;
    
    /**
     * 时间步索引
     */
    private final int timeStep;
    
    /**
     * 构造函数
     */
    public Transition(Observation observation, Action action, double reward,
                     Observation nextObservation, boolean done, int timeStep) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.nextObservation = nextObservation;
        this.done = done;
        this.timeStep = timeStep;
    }
    
    /**
     * 简化构造函数（时间步为0）
     */
    public Transition(Observation observation, Action action, double reward,
                     Observation nextObservation, boolean done) {
        this(observation, action, reward, nextObservation, done, 0);
    }
    
    // Getters
    public Observation getObservation() {
        return observation;
    }
    
    public Action getAction() {
        return action;
    }
    
    public double getReward() {
        return reward;
    }
    
    public Observation getNextObservation() {
        return nextObservation;
    }
    
    public boolean isDone() {
        return done;
    }
    
    public int getTimeStep() {
        return timeStep;
    }
    
    @Override
    public String toString() {
        return String.format("Transition{t=%d, reward=%.3f, done=%b}",
            timeStep, reward, done);
    }
}
