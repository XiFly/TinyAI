package io.leavesfly.tinyai.func;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Select、SliceRange和Full操作的测试
 * 
 * @author TinyAI
 */
public class VariableSliceAndFullTest {

    private static final float EPSILON = 1e-6f;

    @Test
    public void testSelectDim0() {
        // 创建 [2, 3] 的张量
        // [[1, 2, 3],
        //  [4, 5, 6]]
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}));
        
        // 选择第0行
        Variable result = x.select(0, 0);
        
        // 应该得到 [3] 形状: [1, 2, 3]
        assertEquals(Shape.of(3), result.getShape());
        assertArrayEquals(new float[]{1, 2, 3}, result.getValue().getArray(), EPSILON);
    }

    @Test
    public void testSelectDim1() {
        // 创建 [2, 3] 的张量
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}));
        
        // 选择第1列
        Variable result = x.select(1, 1);
        
        // 应该得到 [2] 形状: [2, 5]
        assertEquals(Shape.of(2), result.getShape());
        assertArrayEquals(new float[]{2, 5}, result.getValue().getArray(), EPSILON);
    }

    @Test
    public void testSelectNegativeIndex() {
        // 测试负数索引
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}));
        
        // select(0, -1) 应该选择最后一行
        Variable result = x.select(0, -1);
        
        assertEquals(Shape.of(3), result.getShape());
        assertArrayEquals(new float[]{4, 5, 6}, result.getValue().getArray(), EPSILON);
    }

    @Test
    public void testSelectBackward() {
        // 测试梯度反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}}));
        
        Variable result = x.select(0, 0);  // 选择第0行, 得到 [3]
        // 对result直接计算sum而不是通过Variable的sum,因为result已经是[3]的形状
        Variable loss = result.mul(new Variable(NdArray.of(new float[]{1, 1, 1}))).sum();
        
        loss.backward();
        
        // 梯度应该只在第0行为1,其他位置为0
        float[][] expectedGrad = {{1, 1, 1}, {0, 0, 0}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testSliceRange() {
        // 创建 [2, 5] 的张量
        // [[1, 2, 3, 4, 5],
        //  [6, 7, 8, 9, 10]]
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}));
        
        // 切片 [1:4] 在第1维
        Variable result = x.sliceRange(1, 1, 4);
        
        // 应该得到 [2, 3] 形状
        assertEquals(Shape.of(2, 3), result.getShape());
        assertArrayEquals(new float[][]{{2, 3, 4}, {7, 8, 9}}, result.getValue().getMatrix());
    }

    @Test
    public void testSliceRangeNegativeIndices() {
        // 测试负数索引
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}));
        
        // sliceRange(1, -3, -1) 应该切片倒数第3到倒数第1个元素(不包含-1)
        // -3 -> 5-3=2, -1 -> 5-1+1=5 (end不包含,所以实际是到4)
        Variable result = x.sliceRange(1, -3, -1);
        
        assertEquals(Shape.of(2, 2), result.getShape());
        assertArrayEquals(new float[][]{{3, 4}, {8, 9}}, result.getValue().getMatrix());
    }

    @Test
    public void testSliceRangeBackward() {
        // 测试梯度反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}}));
        
        Variable result = x.sliceRange(1, 1, 3);  // 切片 [1:3]
        Variable loss = result.sum();
        
        loss.backward();
        
        // 梯度应该只在 [1:3] 范围内为1,其他位置为0
        float[][] expectedGrad = {{0, 1, 1, 0}, {0, 1, 1, 0}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

    @Test
    public void testFull() {
        // 创建 [2, 3] 形状,值为5.0的张量
        Variable x = Variable.full(Shape.of(2, 3), 5.0f);
        
        assertEquals(Shape.of(2, 3), x.getShape());
        
        float[][] expected = {{5, 5, 5}, {5, 5, 5}};
        assertArrayEquals(expected, x.getValue().getMatrix());
    }

    @Test
    public void testFullWithDifferentShapes() {
        // 测试不同形状
        Variable x1 = Variable.full(Shape.of(3), 1.0f);
        assertEquals(Shape.of(3), x1.getShape());
        assertArrayEquals(new float[]{1, 1, 1}, x1.getValue().getArray(), EPSILON);
        
        Variable x2 = Variable.full(Shape.of(2, 2), 7.5f);
        assertEquals(Shape.of(2, 2), x2.getShape());
        assertArrayEquals(new float[][]{{7.5f, 7.5f}, {7.5f, 7.5f}}, x2.getValue().getMatrix());
    }

    @Test
    public void testSelect3D() {
        // 测试3D张量的选择
        // 创建 [2, 3, 4] 的张量
        float[][][] data3d = {
            {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
            {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
        };
        Variable x = new Variable(NdArray.of(data3d));
        
        // 选择第0个批次
        Variable result = x.select(0, 0);
        
        // 应该得到 [3, 4] 形状
        assertEquals(Shape.of(3, 4), result.getShape());
        assertArrayEquals(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, 
                result.getValue().getMatrix());
    }

    @Test
    public void testSliceRange3D() {
        // 测试3D张量的切片
        float[][][] data3d = {
            {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
            {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
        };
        Variable x = new Variable(NdArray.of(data3d));
        
        // 在维度2上切片 [1:3]
        Variable result = x.sliceRange(2, 1, 3);
        
        // 应该得到 [2, 3, 2] 形状
        assertEquals(Shape.of(2, 3, 2), result.getShape());
    }

    @Test
    public void testSelectAndSliceRangeCombined() {
        // 组合使用select和sliceRange
        Variable x = new Variable(NdArray.of(new float[][]{            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15}
        }));
        
        // 先切片行 [0:2],再选择第2列
        Variable sliced = x.sliceRange(0, 0, 2);  // 得到 [2, 5]
        Variable selected = sliced.select(1, 2);  // 得到 [2]
        
        assertEquals(Shape.of(2), selected.getShape());
        assertArrayEquals(new float[]{3, 8}, selected.getValue().getArray(), EPSILON);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testSelectOutOfBounds() {
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}));
        x.select(0, 5);  // 应该抛出异常
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSelectInvalidDim() {
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2}, {3, 4}}));
        x.select(5, 0);  // 应该抛出异常
    }

    @Test
    public void testSplit() {
        // 测试基本split功能
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}));
        
        // 在第1维上分割,每块大小为2
        Variable[] splits = x.split(2, 1);
        
        // 应该得到3个分块: [2,2], [2,2], [2,1]
        assertEquals(3, splits.length);
        
        assertEquals(Shape.of(2, 2), splits[0].getShape());
        assertArrayEquals(new float[][]{{1, 2}, {6, 7}}, splits[0].getValue().getMatrix());
        
        assertEquals(Shape.of(2, 2), splits[1].getShape());
        assertArrayEquals(new float[][]{{3, 4}, {8, 9}}, splits[1].getValue().getMatrix());
        
        assertEquals(Shape.of(2, 1), splits[2].getShape());
        assertArrayEquals(new float[][]{{5}, {10}}, splits[2].getValue().getMatrix());
    }

    @Test
    public void testSplitBackward() {
        // 测试split的梯度反向传播
        Variable x = new Variable(NdArray.of(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}}));
        
        Variable[] splits = x.split(2, 1);  // 分成 [2,2] 和 [2,2]
        
        // 对第一个分块计算损失
        Variable loss = splits[0].sum();
        loss.backward();
        
        // 梯度应该只在第一个分块的位置为1,其他位置为0
        float[][] expectedGrad = {{1, 1, 0, 0}, {1, 1, 0, 0}};
        assertArrayEquals(expectedGrad, x.getGrad().getMatrix());
    }

}
