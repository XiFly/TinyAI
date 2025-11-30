package io.leavesfly.tinyai.wm.env;

import io.leavesfly.tinyai.wm.model.Action;
import io.leavesfly.tinyai.wm.model.Observation;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 简单驾驶环境
 * 一个简化的2D驾驶模拟环境，用于测试世界模型
 * 
 * 状态空间：
 * - 位置 (x, y)
 * - 速度 (vx, vy)
 * - 朝向角度 θ
 * 
 * 动作空间：
 * - 转向 [-1, 1]
 * - 加速 [-1, 1]
 * - 刹车 [0, 1]
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class SimpleDrivingEnvironment implements Environment {
    
    /**
     * 车辆状态
     */
    private double x, y;          // 位置
    private double vx, vy;        // 速度
    private double theta;         // 朝向角度
    
    /**
     * 环境参数
     */
    private final double maxSpeed = 20.0;
    private final double maxAcceleration = 3.0;
    private final double dt = 0.1; // 时间步长
    
    /**
     * 目标位置
     */
    private double targetX = 100.0;
    private double targetY = 0.0;
    
    /**
     * 步数计数器
     */
    private int stepCount;
    private final int maxSteps = 1000;
    
    /**
     * 观察和动作维度
     */
    private static final int OBSERVATION_SIZE = 8;  // [x, y, vx, vy, theta, target_x, target_y, dist]
    private static final int ACTION_SIZE = 3;        // [steering, throttle, brake]
    
    @Override
    public Observation reset() {
        // 重置车辆到初始位置
        this.x = 0.0;
        this.y = 0.0;
        this.vx = 0.0;
        this.vy = 0.0;
        this.theta = 0.0;
        this.stepCount = 0;
        
        // 随机目标位置
        this.targetX = 80.0 + Math.random() * 40.0;
        this.targetY = -20.0 + Math.random() * 40.0;
        
        return getCurrentObservation();
    }
    
    @Override
    public StepResult step(Action action) {
        stepCount++;
        
        // 1. 解析动作
        double[] actionValues = action.toArray();
        double steering = Math.max(-1.0, Math.min(1.0, actionValues[0]));
        double throttle = Math.max(-1.0, Math.min(1.0, actionValues[1]));
        double brake = actionValues.length > 2 ? Math.max(0.0, Math.min(1.0, actionValues[2])) : 0.0;
        
        // 2. 更新朝向
        theta += steering * 0.1; // 转向影响朝向
        
        // 3. 计算加速度
        double acceleration = throttle * maxAcceleration - brake * maxAcceleration * 2.0;
        double ax = acceleration * Math.cos(theta);
        double ay = acceleration * Math.sin(theta);
        
        // 4. 更新速度
        vx += ax * dt;
        vy += ay * dt;
        
        // 限制最大速度
        double speed = Math.sqrt(vx * vx + vy * vy);
        if (speed > maxSpeed) {
            vx = vx / speed * maxSpeed;
            vy = vy / speed * maxSpeed;
        }
        
        // 5. 更新位置
        x += vx * dt;
        y += vy * dt;
        
        // 6. 计算奖励
        double reward = calculateReward(steering, throttle, brake);
        
        // 7. 检查终止条件
        boolean done = checkTermination();
        String info = done ? getTerminationReason() : "running";
        
        // 8. 返回结果
        Observation observation = getCurrentObservation();
        return new StepResult(observation, reward, done, info);
    }
    
    /**
     * 获取当前观察
     */
    private Observation getCurrentObservation() {
        // 计算到目标的距离
        double distToTarget = Math.sqrt(
            Math.pow(targetX - x, 2) + Math.pow(targetY - y, 2)
        );
        
        // 状态向量
        double[] stateValues = {
            x / 100.0,           // 归一化位置
            y / 100.0,
            vx / maxSpeed,       // 归一化速度
            vy / maxSpeed,
            theta / Math.PI,     // 归一化角度
            targetX / 100.0,     // 归一化目标
            targetY / 100.0,
            distToTarget / 100.0 // 归一化距离
        };
        
        NdArray stateVector = NdArray.of(floatArray(stateValues), io.leavesfly.tinyai.ndarr.Shape.of(stateValues.length));
        
        // 简化的视觉观察（这里用零向量代替真实图像）
        NdArray visualObservation = NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(3, 64, 64));
        
        return new Observation(visualObservation, stateVector);
    }
    
    /**
     * 计算奖励
     */
    private double calculateReward(double steering, double throttle, double brake) {
        // 1. 距离奖励（越靠近目标越高）
        double distToTarget = Math.sqrt(
            Math.pow(targetX - x, 2) + Math.pow(targetY - y, 2)
        );
        double distReward = -distToTarget * 0.01;
        
        // 2. 速度奖励（鼓励保持合理速度）
        double speed = Math.sqrt(vx * vx + vy * vy);
        double speedReward = -Math.abs(speed - 10.0) * 0.01;
        
        // 3. 动作惩罚（鼓励平滑控制）
        double actionPenalty = -(Math.abs(steering) + Math.abs(throttle) + brake) * 0.001;
        
        // 4. 到达目标奖励
        double reachReward = 0.0;
        if (distToTarget < 5.0) {
            reachReward = 100.0;
        }
        
        return distReward + speedReward + actionPenalty + reachReward;
    }
    
    /**
     * 检查终止条件
     */
    private boolean checkTermination() {
        // 1. 到达目标
        double distToTarget = Math.sqrt(
            Math.pow(targetX - x, 2) + Math.pow(targetY - y, 2)
        );
        if (distToTarget < 5.0) {
            return true;
        }
        
        // 2. 超出边界
        if (Math.abs(x) > 200 || Math.abs(y) > 200) {
            return true;
        }
        
        // 3. 超过最大步数
        return stepCount >= maxSteps;
    }
    
    /**
     * 获取终止原因
     */
    private String getTerminationReason() {
        double distToTarget = Math.sqrt(
            Math.pow(targetX - x, 2) + Math.pow(targetY - y, 2)
        );
        
        if (distToTarget < 5.0) {
            return "target_reached";
        } else if (Math.abs(x) > 200 || Math.abs(y) > 200) {
            return "out_of_bounds";
        } else if (stepCount >= maxSteps) {
            return "max_steps";
        }
        return "unknown";
    }
    
    @Override
    public void close() {
        // 清理资源（这里无需特别清理）
    }
    
    @Override
    public int getActionSize() {
        return ACTION_SIZE;
    }
    
    @Override
    public int getObservationSize() {
        return OBSERVATION_SIZE;
    }
    
    // Getters for debugging
    public double getX() { return x; }
    public double getY() { return y; }
    public double getVx() { return vx; }
    public double getVy() { return vy; }
    public double getTheta() { return theta; }
    public double getTargetX() { return targetX; }
    public double getTargetY() { return targetY; }
    public int getStepCount() { return stepCount; }
    
    // 辅助方法：double[]转float[]
    private float[] floatArray(double[] arr) {
        float[] result = new float[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = (float) arr[i];
        }
        return result;
    }
}
