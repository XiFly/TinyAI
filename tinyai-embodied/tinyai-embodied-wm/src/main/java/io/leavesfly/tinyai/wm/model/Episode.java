package io.leavesfly.tinyai.wm.model;

import java.util.ArrayList;
import java.util.List;

/**
 * 情景（Episode）
 * 表示一个完整的智能体交互序列
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class Episode {
    
    /**
     * 转换列表
     */
    private final List<Transition> transitions;
    
    /**
     * 情景ID
     */
    private final String episodeId;
    
    /**
     * 开始时间戳
     */
    private final long startTime;
    
    /**
     * 结束时间戳
     */
    private long endTime;
    
    /**
     * 是否完成
     */
    private boolean completed;
    
    /**
     * 构造函数
     */
    public Episode(String episodeId) {
        this.episodeId = episodeId;
        this.transitions = new ArrayList<>();
        this.startTime = System.currentTimeMillis();
        this.completed = false;
    }
    
    /**
     * 添加转换
     */
    public void addTransition(Transition transition) {
        transitions.add(transition);
        if (transition.isDone()) {
            completed = true;
            endTime = System.currentTimeMillis();
        }
    }
    
    /**
     * 获取情景长度
     */
    public int getLength() {
        return transitions.size();
    }
    
    /**
     * 获取总奖励
     */
    public double getTotalReward() {
        return transitions.stream()
            .mapToDouble(Transition::getReward)
            .sum();
    }
    
    /**
     * 获取平均奖励
     */
    public double getAverageReward() {
        if (transitions.isEmpty()) {
            return 0.0;
        }
        return getTotalReward() / transitions.size();
    }
    
    /**
     * 获取持续时间（毫秒）
     */
    public long getDuration() {
        if (completed) {
            return endTime - startTime;
        }
        return System.currentTimeMillis() - startTime;
    }
    
    /**
     * 获取指定索引的转换
     */
    public Transition getTransition(int index) {
        if (index < 0 || index >= transitions.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + transitions.size());
        }
        return transitions.get(index);
    }
    
    // Getters
    public List<Transition> getTransitions() {
        return new ArrayList<>(transitions);
    }
    
    public String getEpisodeId() {
        return episodeId;
    }
    
    public long getStartTime() {
        return startTime;
    }
    
    public long getEndTime() {
        return endTime;
    }
    
    public boolean isCompleted() {
        return completed;
    }
    
    @Override
    public String toString() {
        return String.format("Episode{id=%s, length=%d, totalReward=%.3f, completed=%b}",
            episodeId, getLength(), getTotalReward(), completed);
    }
}
