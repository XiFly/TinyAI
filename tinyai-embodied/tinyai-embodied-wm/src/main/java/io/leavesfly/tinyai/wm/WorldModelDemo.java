package io.leavesfly.tinyai.wm;

import io.leavesfly.tinyai.wm.core.WorldModel;
import io.leavesfly.tinyai.wm.env.Environment;
import io.leavesfly.tinyai.wm.env.SimpleDrivingEnvironment;
import io.leavesfly.tinyai.wm.model.Episode;

/**
 * 世界模型演示程序
 * 展示世界模型智能体的基本使用方法
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class WorldModelDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("TinyAI 世界模型具身智能演示");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 演示1：基本环境交互
        demoBasicInteraction();
        
        System.out.println();
        
        // 演示2：完整情景运行
        demoEpisodeRun();
        
        System.out.println();
        
        // 演示3：想象训练
        demoDreamTraining();
        
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("演示完成！");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 演示1：基本环境交互
     */
    private static void demoBasicInteraction() {
        System.out.println("【演示1：基本环境交互】");
        System.out.println("-".repeat(80));
        
        // 1. 创建环境
        Environment env = new SimpleDrivingEnvironment();
        System.out.println("✓ 创建简单驾驶环境");
        System.out.println("  - 观察空间维度: " + env.getObservationSize());
        System.out.println("  - 动作空间维度: " + env.getActionSize());
        
        // 2. 创建世界模型配置
        WorldModel.WorldModelConfig config = WorldModel.WorldModelConfig.createDefault();
        System.out.println("✓ 创建世界模型配置");
        System.out.println("  - 潜在空间维度: " + config.getLatentSize());
        System.out.println("  - 隐藏状态维度: " + config.getHiddenSize());
        System.out.println("  - 高斯混合数: " + config.getNumMixtures());
        
        // 3. 创建世界模型
        WorldModel worldModel = new WorldModel(config);
        System.out.println("✓ 创建世界模型");
        System.out.println("  - VAE编码器: 已初始化");
        System.out.println("  - MDN-RNN: 已初始化");
        System.out.println("  - 控制器: 已初始化");
        
        // 4. 创建智能体
        WorldModelAgent agent = new WorldModelAgent(worldModel, env);
        System.out.println("✓ 创建世界模型智能体");
        
        // 5. 运行几步
        System.out.println("\n运行10步交互...");
        agent.reset();
        
        for (int i = 0; i < 10; i++) {
            Environment.StepResult result = agent.step();
            System.out.printf("  步骤 %d: 奖励=%.3f, 完成=%b\n", 
                i + 1, result.getReward(), result.isDone());
            
            if (result.isDone()) {
                System.out.println("  提前终止: " + result.getInfo());
                break;
            }
        }
        
        System.out.println("\n" + agent.getStatistics());
    }
    
    /**
     * 演示2：完整情景运行
     */
    private static void demoEpisodeRun() {
        System.out.println("【演示2：完整情景运行】");
        System.out.println("-".repeat(80));
        
        // 创建环境和智能体
        Environment env = new SimpleDrivingEnvironment();
        WorldModel.WorldModelConfig config = new WorldModel.WorldModelConfig(
            8,    // observationSize
            32,   // latentSize
            128,  // hiddenSize
            3,    // actionSize
            64,   // vaeHiddenSize
            5,    // numMixtures
            false // deterministic
        );
        WorldModel worldModel = new WorldModel(config);
        WorldModelAgent agent = new WorldModelAgent(worldModel, env);
        
        System.out.println("运行3个情景...\n");
        
        for (int i = 0; i < 3; i++) {
            System.out.println("情景 " + (i + 1) + ":");
            Episode episode = agent.runEpisode(100);
            
            System.out.println("  - 长度: " + episode.getLength());
            System.out.println("  - 总奖励: " + String.format("%.3f", episode.getTotalReward()));
            System.out.println("  - 平均奖励: " + String.format("%.3f", episode.getAverageReward()));
            System.out.println("  - 状态: " + (episode.isCompleted() ? "完成" : "进行中"));
            System.out.println();
        }
    }
    
    /**
     * 演示3：想象训练
     */
    private static void demoDreamTraining() {
        System.out.println("【演示3：在想象环境中训练】");
        System.out.println("-".repeat(80));
        
        // 创建环境和智能体
        Environment env = new SimpleDrivingEnvironment();
        WorldModel worldModel = new WorldModel(WorldModel.WorldModelConfig.createDefault());
        WorldModelAgent agent = new WorldModelAgent(worldModel, env);
        
        // 先在真实环境中运行一小段，收集初始经验
        System.out.println("在真实环境中收集初始经验...");
        agent.reset();
        for (int i = 0; i < 20; i++) {
            agent.step();
        }
        System.out.println("✓ 收集了20步经验\n");
        
        // 在想象环境中训练
        System.out.println("在想象环境中进行rollout...");
        Episode dreamEpisode = agent.trainInDream(50);
        
        System.out.println("✓ 想象训练完成");
        System.out.println("  - 想象情景长度: " + dreamEpisode.getLength());
        System.out.println("  - 想象总奖励: " + String.format("%.3f", dreamEpisode.getTotalReward()));
        System.out.println("  - 想象平均奖励: " + String.format("%.3f", dreamEpisode.getAverageReward()));
        
        System.out.println("\n说明：");
        System.out.println("  想象训练允许智能体在内部模型中进行规划和学习，");
        System.out.println("  而无需与真实环境交互，大大提高了样本效率。");
    }
}
