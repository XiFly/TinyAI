package io.leavesfly.tinyai.wm.model;

/**
 * 动作类型枚举
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public enum ActionType {
    /**
     * 连续动作空间（如转向角度、加速度）
     */
    CONTINUOUS,
    
    /**
     * 离散动作空间（如前进、后退、左转、右转）
     */
    DISCRETE
}
