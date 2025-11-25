package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * VLA数据模型测试
 *
 * @author TinyAI
 */
public class VLAModelTest {

    @Test
    public void testVisionInput() {
        float[][][] imageData = new float[64][64][3];
        NdArray rgbImage = NdArray.of(imageData);

        VisionInput visionInput = new VisionInput(rgbImage);

        assertNotNull(visionInput);
        assertNotNull(visionInput.getRgbImage());
        assertEquals(64, visionInput.getRgbImage().getShape().getShapeDims()[0]);
        assertEquals(64, visionInput.getRgbImage().getShape().getShapeDims()[1]);
        assertEquals(3, visionInput.getRgbImage().getShape().getShapeDims()[2]);
        assertTrue(visionInput.getTimestamp() > 0);
    }

    @Test
    public void testLanguageInput() {
        String instruction = "Pick up the red cube";
        LanguageInput languageInput = new LanguageInput(instruction);

        assertNotNull(languageInput);
        assertEquals(instruction, languageInput.getInstruction());
    }

    @Test
    public void testProprioceptionInput() {
        float[] jointPositions = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
        float[] jointVelocities = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        NdArray positions = NdArray.of(jointPositions);
        NdArray velocities = NdArray.of(jointVelocities);

        ProprioceptionInput proprioInput = new ProprioceptionInput(positions, velocities);

        assertNotNull(proprioInput);
        assertNotNull(proprioInput.getJointPositions());
        assertNotNull(proprioInput.getJointVelocities());
    }

    @Test
    public void testVLAState() {
        float[][][] imageData = new float[64][64][3];
        VisionInput visionInput = new VisionInput(NdArray.of(imageData));
        LanguageInput languageInput = new LanguageInput("Test instruction");

        VLAState state = new VLAState(visionInput, languageInput);

        assertNotNull(state);
        assertNotNull(state.getVisionInput());
        assertNotNull(state.getLanguageInput());
        assertNotNull(state.getAttentionWeights());
        assertTrue(state.getTimestamp() > 0);
    }

    @Test
    public void testVLAAction() {
        float[] actionValues = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
        NdArray continuousAction = NdArray.of(actionValues);

        VLAAction action = new VLAAction(continuousAction, 0, ActionType.MOVE_END_EFFECTOR);

        assertNotNull(action);
        assertNotNull(action.getContinuousAction());
        assertEquals(0, action.getDiscreteAction());
        assertEquals(ActionType.MOVE_END_EFFECTOR, action.getActionType());
        assertEquals(1.0, action.getConfidence());
    }

    @Test
    public void testActionType() {
        assertEquals(7, ActionType.values().length);
        assertEquals("移动末端执行器", ActionType.MOVE_END_EFFECTOR.getDescription());
        assertEquals("抓取物体", ActionType.GRASP_OBJECT.getDescription());
    }

    @Test
    public void testTaskConfig() {
        TaskConfig config = new TaskConfig();
        config.setTaskName("PickAndPlace");
        config.setMaxSteps(100);
        config.setSuccessReward(100.0);

        assertEquals("PickAndPlace", config.getTaskName());
        assertEquals(100, config.getMaxSteps());
        assertEquals(100.0, config.getSuccessReward());
        assertFalse(config.isRender());
    }
}
