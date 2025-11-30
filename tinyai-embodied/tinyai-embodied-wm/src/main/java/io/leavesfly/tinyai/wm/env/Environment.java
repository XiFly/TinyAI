package io.leavesfly.tinyai.wm.env;

import io.leavesfly.tinyai.wm.model.Action;
import io.leavesfly.tinyai.wm.model.Observation;

/**
 * 环境接口
 * 定义智能体与环境交互的标准接口
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public interface Environment {
    
    /**
     * 重置环境到初始状态
     *
     * @return 初始观察
     */
    Observation reset();
    
    /**
     * 执行一个动作
     *
     * @param action 动作
     * @return 环境响应（观察、奖励、是否终止）
     */
    StepResult step(Action action);
    
    /**
     * 关闭环境，释放资源
     */
    void close();
    
    /**
     * 获取动作空间维度
     */
    int getActionSize();
    
    /**
     * 获取观察空间维度
     */
    int getObservationSize();
    
    /**
     * 环境步进结果
     */
    class StepResult {
        private final Observation observation;
        private final double reward;
        private final boolean done;
        private final String info;
        
        public StepResult(Observation observation, double reward, boolean done, String info) {
            this.observation = observation;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }
        
        public Observation getObservation() {
            return observation;
        }
        
        public double getReward() {
            return reward;
        }
        
        public boolean isDone() {
            return done;
        }
        
        public String getInfo() {
            return info;
        }
        
        @Override
        public String toString() {
            return String.format("StepResult{reward=%.3f, done=%b, info='%s'}",
                reward, done, info);
        }
    }
}
