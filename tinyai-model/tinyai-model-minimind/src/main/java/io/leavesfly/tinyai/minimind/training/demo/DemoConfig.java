package io.leavesfly.tinyai.minimind.training.demo;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.io.*;
import java.util.*;

/**
 * MiniMind 训练演示 - 配置与工具类
 * 
 * 集中管理：
 * - 路径常量
 * - 共享状态（分词器）
 * - 模型配置
 * - 文件读写工具
 * 
 * @author TinyAI Team
 */
public class DemoConfig {

    // ========== 路径常量 ==========
    
    public static final String DATA_DIR = "./data/minimind_training";
    public static final String CHECKPOINT_DIR = "./checkpoints/minimind";

    // ========== 共享状态 ==========
    
    /** 共享分词器 - 全阶段复用 */
    private static MiniMindTokenizer sharedTokenizer;

    public static MiniMindTokenizer getSharedTokenizer() {
        return sharedTokenizer;
    }

    public static void setSharedTokenizer(MiniMindTokenizer tokenizer) {
        sharedTokenizer = tokenizer;
    }

    // ========== 模型配置 ==========

    /**
     * 创建超小型配置（用于快速演示）
     */
    public static MiniMindConfig createMicroConfig(int vocabSize) {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLen(64);          // 序列长度
        config.setHiddenSize(128);        // 隐藏维度
        config.setNumLayers(2);           // 层数
        config.setNumHeads(4);            // 注意力头数
        config.setFfnHiddenSize(256);     // FFN隐藏维度
        config.setDropout(0.1f);
        config.setEpsilon(1e-5f);
        return config;
    }

    // ========== 文件工具 ==========

    /**
     * 从文件读取文本行
     */
    public static List<String> readFromFile(String filePath) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }

    /**
     * 写入文件
     */
    public static void writeToFile(List<String> lines, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        }
    }

    // ========== 辅助方法 ==========

    /**
     * int[] 转 List<Integer>
     */
    public static List<Integer> intArrayToList(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int value : array) {
            list.add(value);
        }
        return list;
    }

    /**
     * 提取奖励值
     */
    public static float extractReward(String text) {
        if (text.startsWith("[REWARD:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 0) {
                String rewardStr = text.substring(8, endIdx);
                try {
                    return Float.parseFloat(rewardStr);
                } catch (NumberFormatException e) {
                    return 0.5f;
                }
            }
        }
        return 0.5f;
    }

    /**
     * 移除奖励标签
     */
    public static String removeRewardLabel(String text) {
        return text.replaceFirst("^\\[REWARD:[0-9.]+\\]\\s*", "");
    }
}
