package io.leavesfly.tinyai.wm.training;

/**
 * 训练配置类
 * 定义训练过程的所有超参数
 *
 * @author leavesfly
 * @since 2025-11-04
 */
public class TrainingConfig {
    
    // 数据收集配置
    private int numRandomEpisodes = 50;        // 随机探索情景数
    private int maxStepsPerEpisode = 200;      // 每个情景最大步数
    private int maxBufferSize = 100000;        // 经验缓冲区最大大小
    
    // VAE训练配置
    private int vaeEpochs = 50;                // VAE训练轮数
    private int batchSize = 64;                // 批次大小
    private double vaeLearningRate = 0.001;    // VAE学习率
    
    // MDN-RNN训练配置
    private int rnnEpochs = 50;                // RNN训练轮数
    private double rnnLearningRate = 0.001;    // RNN学习率
    
    // Controller训练配置
    private int controllerEpochs = 100;        // 控制器训练轮数
    private int dreamRolloutsPerEpoch = 16;    // 每轮想象rollout次数
    private int dreamSteps = 50;               // 每次想象的步数
    private double controllerLearningRate = 0.01; // 控制器学习率
    
    // 微调配置
    private int fineTuneEpisodes = 100;        // 真实环境微调情景数
    
    /**
     * 创建默认配置
     */
    public static TrainingConfig createDefault() {
        return new TrainingConfig();
    }
    
    /**
     * 创建快速测试配置（用于开发调试）
     */
    public static TrainingConfig createQuickTest() {
        TrainingConfig config = new TrainingConfig();
        config.numRandomEpisodes = 5;
        config.vaeEpochs = 10;
        config.rnnEpochs = 10;
        config.controllerEpochs = 20;
        config.fineTuneEpisodes = 10;
        config.dreamRolloutsPerEpoch = 4;
        return config;
    }
    
    /**
     * 创建完整训练配置（用于正式训练）
     */
    public static TrainingConfig createFull() {
        TrainingConfig config = new TrainingConfig();
        config.numRandomEpisodes = 100;
        config.vaeEpochs = 100;
        config.rnnEpochs = 100;
        config.controllerEpochs = 200;
        config.fineTuneEpisodes = 200;
        config.dreamRolloutsPerEpoch = 32;
        return config;
    }
    
    // Getters and Setters
    public int getNumRandomEpisodes() {
        return numRandomEpisodes;
    }
    
    public void setNumRandomEpisodes(int numRandomEpisodes) {
        this.numRandomEpisodes = numRandomEpisodes;
    }
    
    public int getMaxStepsPerEpisode() {
        return maxStepsPerEpisode;
    }
    
    public void setMaxStepsPerEpisode(int maxStepsPerEpisode) {
        this.maxStepsPerEpisode = maxStepsPerEpisode;
    }
    
    public int getMaxBufferSize() {
        return maxBufferSize;
    }
    
    public void setMaxBufferSize(int maxBufferSize) {
        this.maxBufferSize = maxBufferSize;
    }
    
    public int getVaeEpochs() {
        return vaeEpochs;
    }
    
    public void setVaeEpochs(int vaeEpochs) {
        this.vaeEpochs = vaeEpochs;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public double getVaeLearningRate() {
        return vaeLearningRate;
    }
    
    public void setVaeLearningRate(double vaeLearningRate) {
        this.vaeLearningRate = vaeLearningRate;
    }
    
    public int getRnnEpochs() {
        return rnnEpochs;
    }
    
    public void setRnnEpochs(int rnnEpochs) {
        this.rnnEpochs = rnnEpochs;
    }
    
    public double getRnnLearningRate() {
        return rnnLearningRate;
    }
    
    public void setRnnLearningRate(double rnnLearningRate) {
        this.rnnLearningRate = rnnLearningRate;
    }
    
    public int getControllerEpochs() {
        return controllerEpochs;
    }
    
    public void setControllerEpochs(int controllerEpochs) {
        this.controllerEpochs = controllerEpochs;
    }
    
    public int getDreamRolloutsPerEpoch() {
        return dreamRolloutsPerEpoch;
    }
    
    public void setDreamRolloutsPerEpoch(int dreamRolloutsPerEpoch) {
        this.dreamRolloutsPerEpoch = dreamRolloutsPerEpoch;
    }
    
    public int getDreamSteps() {
        return dreamSteps;
    }
    
    public void setDreamSteps(int dreamSteps) {
        this.dreamSteps = dreamSteps;
    }
    
    public double getControllerLearningRate() {
        return controllerLearningRate;
    }
    
    public void setControllerLearningRate(double controllerLearningRate) {
        this.controllerLearningRate = controllerLearningRate;
    }
    
    public int getFineTuneEpisodes() {
        return fineTuneEpisodes;
    }
    
    public void setFineTuneEpisodes(int fineTuneEpisodes) {
        this.fineTuneEpisodes = fineTuneEpisodes;
    }
    
    @Override
    public String toString() {
        return "TrainingConfig{\n" +
                "  数据收集:\n" +
                "    随机情景数=" + numRandomEpisodes + "\n" +
                "    每情景步数=" + maxStepsPerEpisode + "\n" +
                "  VAE训练:\n" +
                "    训练轮数=" + vaeEpochs + "\n" +
                "    批次大小=" + batchSize + "\n" +
                "    学习率=" + vaeLearningRate + "\n" +
                "  RNN训练:\n" +
                "    训练轮数=" + rnnEpochs + "\n" +
                "    学习率=" + rnnLearningRate + "\n" +
                "  控制器训练:\n" +
                "    训练轮数=" + controllerEpochs + "\n" +
                "    每轮rollout=" + dreamRolloutsPerEpoch + "\n" +
                "    想象步数=" + dreamSteps + "\n" +
                "  微调:\n" +
                "    微调情景=" + fineTuneEpisodes + "\n" +
                '}';
    }
}
