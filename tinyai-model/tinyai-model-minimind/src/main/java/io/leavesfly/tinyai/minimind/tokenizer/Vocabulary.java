package io.leavesfly.tinyai.minimind.tokenizer;

import java.io.*;
import java.util.*;

/**
 * 词汇表管理类
 * <p>
 * 功能：
 * - Token 到 ID 的双向映射
 * - 特殊 Token 管理
 * - 词汇表序列化/反序列化
 *
 * @author leavesfly
 * @version 1.0
 */
public class Vocabulary implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 特殊 Token 定义
     */
    public static final String PAD_TOKEN = "<pad>";
    public static final String UNK_TOKEN = "<unk>";
    public static final String BOS_TOKEN = "<|im_start|>";
    public static final String EOS_TOKEN = "<|im_end|>";

    /**
     * Token 到 ID 的映射
     */
    private final Map<String, Integer> tokenToId;

    /**
     * ID 到 Token 的映射
     */
    private final Map<Integer, String> idToToken;

    /**
     * 词汇表大小
     */
    private int vocabSize;

    /**
     * 特殊 Token ID
     */
    private final int padTokenId;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;

    /**
     * 构造词汇表
     *
     * @param vocabSize 词汇表大小
     */
    public Vocabulary(int vocabSize) {
        this.vocabSize = vocabSize;
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();

        // 添加特殊 Token (固定 ID)
        this.padTokenId = addToken(PAD_TOKEN, 0);
        this.unkTokenId = addToken(UNK_TOKEN, 1);
        this.bosTokenId = addToken(BOS_TOKEN, 2);
        this.eosTokenId = addToken(EOS_TOKEN, 3);
    }

    /**
     * 从已有映射构造词汇表
     *
     * @param tokenToId Token 到 ID 映射
     */
    public Vocabulary(Map<String, Integer> tokenToId) {
        this.tokenToId = new HashMap<>(tokenToId);
        this.idToToken = new HashMap<>();
        
        // 构建反向映射
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            idToToken.put(entry.getValue(), entry.getKey());
        }
        
        this.vocabSize = tokenToId.size();
        this.padTokenId = tokenToId.getOrDefault(PAD_TOKEN, 0);
        this.unkTokenId = tokenToId.getOrDefault(UNK_TOKEN, 1);
        this.bosTokenId = tokenToId.getOrDefault(BOS_TOKEN, 2);
        this.eosTokenId = tokenToId.getOrDefault(EOS_TOKEN, 3);
    }

    /**
     * 添加 Token
     *
     * @param token Token 字符串
     * @return Token ID
     */
    public int addToken(String token) {
        if (tokenToId.containsKey(token)) {
            return tokenToId.get(token);
        }
        
        int id = tokenToId.size();
        return addToken(token, id);
    }

    /**
     * 添加 Token (指定 ID)
     *
     * @param token Token 字符串
     * @param id    Token ID
     * @return Token ID
     */
    private int addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.put(id, token);
        return id;
    }

    /**
     * 获取 Token 的 ID
     *
     * @param token Token 字符串
     * @return Token ID,如果不存在返回 UNK_TOKEN_ID
     */
    public int getTokenId(String token) {
        return tokenToId.getOrDefault(token, unkTokenId);
    }

    /**
     * 获取 ID 对应的 Token
     *
     * @param id Token ID
     * @return Token 字符串,如果不存在返回 UNK_TOKEN
     */
    public String getToken(int id) {
        return idToToken.getOrDefault(id, UNK_TOKEN);
    }

    /**
     * 检查 Token 是否存在
     */
    public boolean containsToken(String token) {
        return tokenToId.containsKey(token);
    }

    /**
     * 检查 ID 是否存在
     */
    public boolean containsId(int id) {
        return idToToken.containsKey(id);
    }

    /**
     * 获取词汇表大小（实际添加的 token 数量）
     */
    public int getVocabSize() {
        return tokenToId.size();
    }
    
    /**
     * 获取配置的词汇表大小（用于模型配置）
     */
    public int getConfiguredVocabSize() {
        return vocabSize;
    }

    /**
     * 获取特殊 Token ID
     */
    public int getPadTokenId() {
        return padTokenId;
    }

    public int getUnkTokenId() {
        return unkTokenId;
    }

    public int getBosTokenId() {
        return bosTokenId;
    }

    public int getEosTokenId() {
        return eosTokenId;
    }

    /**
     * 获取所有 Token
     */
    public Set<String> getAllTokens() {
        return new HashSet<>(tokenToId.keySet());
    }

    /**
     * 保存词汇表到文件
     *
     * @param filePath 文件路径
     */
    public void save(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // 按 ID 排序
            List<Map.Entry<Integer, String>> sorted = new ArrayList<>(idToToken.entrySet());
            sorted.sort(Map.Entry.comparingByKey());
            
            for (Map.Entry<Integer, String> entry : sorted) {
                writer.write(entry.getValue());
                writer.newLine();
            }
        }
    }

    /**
     * 从文件加载词汇表
     *
     * @param filePath 文件路径
     * @return 词汇表实例
     */
    public static Vocabulary load(String filePath) throws IOException {
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int id = 0;
            while ((line = reader.readLine()) != null) {
                tokenToId.put(line, id++);
            }
        }
        
        return new Vocabulary(tokenToId);
    }

    @Override
    public String toString() {
        return "Vocabulary{" +
                "vocabSize=" + vocabSize +
                ", padTokenId=" + padTokenId +
                ", unkTokenId=" + unkTokenId +
                ", bosTokenId=" + bosTokenId +
                ", eosTokenId=" + eosTokenId +
                '}';
    }
}
