package io.leavesfly.tinyai.vla;

import io.leavesfly.tinyai.agent.vla.model.*;
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
