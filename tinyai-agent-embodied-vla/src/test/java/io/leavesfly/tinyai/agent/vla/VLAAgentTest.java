package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.agent.vla.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * VLA智能体集成测试
 *
 * @author TinyAI
 */
public class VLAAgentTest {

    @Test
    public void testAgentCreation() {
        VLAAgent agent = new VLAAgent(768, 8, 6, 7);
        assertNotNull(agent);
        assertTrue(agent.getParameterCount() > 0);
    }




}
