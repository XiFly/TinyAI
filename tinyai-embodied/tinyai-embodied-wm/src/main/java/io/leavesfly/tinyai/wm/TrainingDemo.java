package io.leavesfly.tinyai.wm;

import io.leavesfly.tinyai.wm.core.WorldModel;
import io.leavesfly.tinyai.wm.env.Environment;
import io.leavesfly.tinyai.wm.env.SimpleDrivingEnvironment;
import io.leavesfly.tinyai.wm.model.Episode;
import io.leavesfly.tinyai.wm.training.TrainingConfig;
import io.leavesfly.tinyai.wm.training.WorldModelTrainer;

/**
 * 世界模型训练演示
 * 展示完整的训练流程和使用方式
 *
 * @author leavesfly
 * @since 2025-11-04
 */
public class TrainingDemo {
    
    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║   TinyAI 世界模型训练演示程序          ║");
        System.out.println("╚════════════════════════════════════════╝\n");
        
        // 1. 创建环境
        System.out.println("1. 初始化环境...");
        Environment environment = new SimpleDrivingEnvironment();
        System.out.println("   ✓ 简单驾驶环境已创建");
        
        // 2. 创建世界模型配置
        System.out.println("\n2. 配置世界模型...");
        WorldModel.WorldModelConfig wmConfig = WorldModel.WorldModelConfig.createDefault();
        System.out.println("   配置参数:");
        System.out.println("   - 观察维度: " + wmConfig.getObservationSize());
        System.out.println("   - 潜在维度: " + wmConfig.getLatentSize());
        System.out.println("   - 隐藏维度: " + wmConfig.getHiddenSize());
        System.out.println("   - 动作维度: " + wmConfig.getActionSize());
        
        // 3. 创建世界模型
        WorldModel worldModel = new WorldModel(wmConfig);
        System.out.println("   ✓ 世界模型已创建");
        
        // 4. 创建智能体
        System.out.println("\n3. 创建智能体...");
        WorldModelAgent agent = new WorldModelAgent(worldModel, environment);
        System.out.println("   ✓ 智能体已创建");
        
        // 5. 配置训练参数
        System.out.println("\n4. 配置训练参数...");
        TrainingConfig trainingConfig = TrainingConfig.createQuickTest();
        System.out.println(trainingConfig);
        
        // 6. 创建训练器
        System.out.println("\n5. 创建训练器...");
        WorldModelTrainer trainer = new WorldModelTrainer(agent, trainingConfig);
        System.out.println("   ✓ 训练器已创建");
        
        // 7. 开始训练
        System.out.println("\n6. 开始训练流程...\n");
        long startTime = System.currentTimeMillis();
        
        try {
            trainer.train();
        } catch (Exception e) {
            System.err.println("训练过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        
        // 8. 评估训练结果
        System.out.println("\n7. 评估训练结果...");
        System.out.println("   总训练时间: " + String.format("%.2f", duration) + " 秒");
        
        // 运行评估
        System.out.println("\n   运行评估情景...");
        double avgReward = agent.evaluate(5);
        System.out.println("   ✓ 平均评估奖励: " + String.format("%.3f", avgReward));
        
        // 9. 演示世界模型的想象能力
        System.out.println("\n8. 演示想象能力...");
        demonstrateDreaming(agent);
        
        // 10. 打印最终统计
        System.out.println("\n9. 最终统计:");
        System.out.println(agent.getStatistics());
        
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║         训练演示完成！                  ║");
        System.out.println("╚════════════════════════════════════════╝");
    }
    
    /**
     * 演示世界模型的想象（内部模拟）能力
     */
    private static void demonstrateDreaming(WorldModelAgent agent) {
        System.out.println("   在想象环境中进行rollout...");
        
        // 重置到一个新状态
        agent.reset();
        
        // 在想象中运行
        for (int i = 0; i < 3; i++) {
            Episode dreamEpisode =
                agent.trainInDream(20);
            
            System.out.printf("   想象情景 %d: 步数=%d, 总奖励=%.3f\n",
                i + 1,
                dreamEpisode.getLength(),
                dreamEpisode.getTotalReward());
        }
        
        System.out.println("   ✓ 想象演示完成");
    }
    
    /**
     * 完整训练示例（使用完整配置）
     */
    public static void runFullTraining() {
        System.out.println("开始完整训练（这将需要较长时间）...\n");
        
        Environment environment = new SimpleDrivingEnvironment();
        WorldModel.WorldModelConfig wmConfig = WorldModel.WorldModelConfig.createDefault();
        WorldModel worldModel = new WorldModel(wmConfig);
        WorldModelAgent agent = new WorldModelAgent(worldModel, environment);
        
        // 使用完整训练配置
        TrainingConfig fullConfig = TrainingConfig.createFull();
        WorldModelTrainer trainer = new WorldModelTrainer(agent, fullConfig);
        
        trainer.train();
        
        // 最终评估
        System.out.println("\n最终评估（20个情景）:");
        double finalReward = agent.evaluate(20);
        System.out.printf("平均奖励: %.3f\n", finalReward);
    }
}
