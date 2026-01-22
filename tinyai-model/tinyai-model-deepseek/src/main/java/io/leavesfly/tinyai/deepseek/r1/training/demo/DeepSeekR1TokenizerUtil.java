package io.leavesfly.tinyai.deepseek.r1.training.demo;

import java.util.*;

/**
 * DeepSeek-R1简单分词器工具类
 * 提供文本编码和解码功能，支持词汇表构建和冻结
 * 
 * @author leavesfly
 */
public class DeepSeekR1TokenizerUtil {
    
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private int nextId;
    private boolean frozen;
    
    public static final int PAD_TOKEN_ID = 0;
    
    public DeepSeekR1TokenizerUtil() {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        this.nextId = 1;
        this.frozen = false;
        // 预注册PAD token
        this.vocab.put("<PAD>", PAD_TOKEN_ID);
        this.reverseVocab.put(PAD_TOKEN_ID, "<PAD>");
    }
    
    /**
     * 编码文本为token ID序列
     */
    public List<Integer> encode(String text) {
        String[] words = text.toLowerCase()
            .replaceAll("[^a-z0-9\\s]", " ")
            .split("\\s+");
        
        List<Integer> tokens = new ArrayList<>();
        for (String word : words) {
            if (word.isEmpty()) continue;
            
            if (!vocab.containsKey(word)) {
                if (!frozen) {
                    vocab.put(word, nextId);
                    reverseVocab.put(nextId, word);
                    nextId++;
                } else {
                    // 冻结后使用UNK token (id=1)
                    tokens.add(1);
                    continue;
                }
            }
            tokens.add(vocab.get(word));
        }
        return tokens;
    }
    
    /**
     * 解码token ID序列为文本
     */
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            if (token == PAD_TOKEN_ID) continue;
            if (reverseVocab.containsKey(token)) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(reverseVocab.get(token));
            }
        }
        return sb.toString();
    }
    
    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return nextId;
    }
    
    /**
     * 冻结词汇表，不再增加新词
     */
    public void freeze() {
        this.frozen = true;
    }
    
    /**
     * 移除文本中的标签（任务类型、奖励等）
     */
    public static String removeLabels(String text) {
        // 移除任务类型标签 [MATH] [LOGIC] [REASONING] [CODING] [REFLECTION]
        // 移除奖励标签 [REWARD:x.xx]
        // 移除验证类型标签 [TYPE:xxx]
        return text.replaceFirst("^\\[REWARD:[\\d.]+\\]\\s*", "")
                   .replaceFirst("^\\[TYPE:\\w+\\]\\s*", "")
                   .replaceFirst("^\\[\\w+\\]\\s*", "");
    }
    
    /**
     * 提取RLHF奖励值
     */
    public static float extractReward(String text) {
        if (text.startsWith("[REWARD:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 8) {
                try {
                    return Float.parseFloat(text.substring(8, endIdx));
                } catch (NumberFormatException e) {
                    return 0.5f;  // 默认中等奖励
                }
            }
        }
        return 0.5f;
    }
    
    /**
     * 提取RLVR验证器类型
     */
    public static String extractVerifierType(String text) {
        if (text.startsWith("[TYPE:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 6) {
                return text.substring(6, endIdx).trim();
            }
        }
        return "math";  // 默认为数学验证器
    }
}
