---
prev-chapter: "Home"
prev-url: "https://rlhfbook.com/"
page-title: Leetcode (Hot 100)
next-chapter: "nanochat"
next-url: "02-cs336-LLM"
---

# Leetcode (Hot 100)

* **题目来源**: [LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/).
* **题解可视化**: 本文所有代码均可直接复制到: [Python Tutor](https://pythontutor.com/render.html#mode=edit) 进行可视化运行.
* [为什么计算机中需要从 0 开始计数?](https://www.cs.utexas.edu/~EWD/transcriptions/EWD08xx/EWD831.html)










## 哈希





### [1. 两数之和](https://leetcode.cn/problems/two-sum)

* **方法**: 哈希表法 -- 定义一个 recorded 哈希表, 即, 一个字典, python 中的字典就是一个高度优化过的哈希表, 这个字典用于存储访问过的元素并记录它们的索引.
* **时间复杂度**: O(n) -- 遍历数组一次, 一共 n 次迭代.
* **空间复杂度**: O(n) -- 哈希表 recorded 最多储存 n 个键值对.

```python
from typing import List
class Solution1:
    def twoSum(self, nums: List[int], target: int):
        recorded = {}
        for i in range(len(nums)):
            if target - nums[i] in recorded:
                return [recorded[target - nums[i]], i]
            recorded[nums[i]] = i

sol = Solution1()
test_cases = [
    ([2, 7, 11, 15], 9, [0, 1]),
    ([3, 2, 4], 6, [1, 2])
]
for i, (nums, target, res) in enumerate(test_cases):
    assert sol.twoSum(nums, target) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```




### [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams)


* **方法**: 哈希表 -- 构建一个以有序词为键, 对应异位词列表为值的字典. 最后返回 values 的 list 即可.
* **时间复杂度**: O(n × m log m) -- n 是字符串数量, m 是字符串平均长度 (排序耗时).
* **空间复杂度**: O(n × m) -- 存储所有字符串的分组结果.

```python
from collections import defaultdict
from typing import List

class Solution49:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = defaultdict(list)
        for st in strs:
            key = ''.join(sorted(st))
            mp[key].append(st)
        return list(mp.values())
    
sol = Solution49()
def sort_res(group):
    return sorted(sorted(strs) for strs in group)

test_cases = [
    (["eat", "tea", "tan", "ate", "nat", "bat"], [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]),
    ([""], [[""]])
]
for i, (strs, res) in enumerate(test_cases):
    assert sort_res(sol.groupAnagrams(strs)) == sorted(res), f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```




## 双指针

### [283. 移动零](https://leetcode.cn/problems/move-zeroes)
* 方法: 双指针 -- 快指针 right 负责遍历数组的每一个元素, 慢指针 left 负责标记下一个非零元素该放置的位置, 注意, left 左侧的所有元素都是已经处理好了的非零元素. 因为, right 处元素和 left 处元素交换的前提是 right 处元素碰到非零元素了.
* 时间复杂度: O(n) -- 仅对数组进行一次遍历.
* 空间复杂度: O(n) -- 算法​​原地修改​​数组, 只使用了固定数量的额外变量 left 和 right.

```python
from typing import List
class Solution283:
    def moveZeroes(self, nums: List[int]) -> None:
        left = 0
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[right], nums[left] = nums[left], nums[right]
                left += 1 

sol = Solution283()
test_cases = [
    ([0, 1, 0, 3, 12], [1, 3, 12, 0, 0]),
    ([0], [0])
]
for i, (nums, res) in enumerate(test_cases):
    sol.moveZeroes(nums)
    assert nums == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```



### [41. 接雨水](https://leetcode.cn/problems/trapping-rain-water)
* 方法: 双指针法 -- 循环条件值得注意, 是 left <= right, 要取等号, 保证遍历到每一个可能位置的雨水高度计算.
* 时间复杂度: O(n) -- 循环内每个元素仅需遍历一次.
* 空间复杂度: O(1) -- 仅仅需要分配 res, left, right, left_max, right_max 几个变量的空间.

```python
from typing import List
class Solution41:
    def trap(self, height: List[int]) -> int:
        res = 0
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        while left <= right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if left_max <= right_max:
                res += left_max - height[left]
                left += 1
            else:
                res += right_max - height[right]
                right -= 1
        return res

sol = Solution41()
test_cases = [
    ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
    ([4,2,0,3,2,5], 9)
]
for i, (input, res) in enumerate(test_cases):
    assert sol.trap(input) == res, f"test case {i + 1} failed."
print("\nAll test cases passed successfully!")
```



## 滑动窗口


### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum)

* 方法: 滑动窗口法 -- 用右指针进行探索, 当找到当前和 (左右指针包围子数组之和) 大于等于目标值的右指针时, 进入右移左指针的循环, 缩小窗口寻找最小全局窗口大小. 因为遍历了所有的 right, 对每一个 right 都找到了局部最优解, 所以最终该方法必然能找到全局最优解.
* 时间复杂度: O(n) -- left 和 right 最多移动 n 次, 虽然内部有 while 循环, 但 left 从头到尾只增不减, 不会回退, 所以整体操作次数是 2n 级别.
* 空间复杂度: O(1) -- 只需要常数个变量 (left, right, current_sum, min_Len) 维护状态



```python
from typing import List
class Solution209:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_len = float("inf")
        left = 0
        cur_sum = 0
        for right in range(len(nums)):
            cur_sum += nums[right]
            while cur_sum >= target:
                min_len = min(min_len, right - left + 1)
                cur_sum -= nums[left]
                left += 1
        return min_len if min_len != float("inf") else 0

sol = Solution209()
test_cases = [
    (7, [2,3,1,2,4,3], 2),
    (4, [1,4,4], 1),
    (11, [1,1,1,1,1,1,1,1], 0)
]
for i, (target, nums, res) in enumerate(test_cases):
    assert sol.minSubArrayLen(target, nums) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```




## 普通数组


### [56. 合并区间](https://leetcode.cn/problems/merge-intervals)

* 方法: 排序 + 贪心合并 -- 首先根据首元素排序, 随后每一步都选择最优, 即合并进前一个区间.
* 时间复杂度: O(N log N) -- Timsort 算法, 最坏和平均时间复杂度均为 O(N log N).
* 空间复杂度: O(N).

```python
from typing import List
class Solution56:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        merged = []
        merged.sort(key = lambda x: x[0])
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged

sol = Solution56()
test_cases = [
    ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
    ([[1, 4], [4, 5]], [[1, 5]])
]
for i, (intervals, res) in enumerate(test_cases):
    assert sol.merge(intervals) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```



### [189. 轮转数组](https://leetcode.cn/problems/rotate-array)

* 方法: 三次轮转法 -- 向右轮转数组 k 次等价于整体翻转数组, 然后再依次翻转前 k % n 个数组成的子数组, 和剩下的数组.
* 时间复杂度: O(n) -- 就是执行了三次翻转的操作, 每一次翻转就是遍历大半个数组长度次, 做交换操作.
* 空间复杂度: O(1) -- 仅仅引入了 i, j, n, k 几个变量, 对于 nums 的操作是在原地进行的.

```python
from typing import List

class Solution189:
    def rotate(self, nums: List[int], k: int) -> None:
        def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        n = len(nums)
        k %= n
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)

sol = Solution189()
test_cases = [
    ([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]),
    ([-1, -100, 3, 99], 2, [3, 99, -1, -100])
]
for i, (nums, k, res) in enumerate(test_cases):
    sol.rotate(nums, k)
    assert nums == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```


## 链表

### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/description)

* 方法: 迭代法 -- 经典方法, 必背. 参考灵山的视频.
* 时间复杂度: O(n) -- 每个节点仅访问一次.
* 空间复杂度: O(1) -- 仅仅使用了常数个额外指针变量 pre, cur, nxt.

```python
from typing import Optional, List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_linked_list(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for i in range(1, len(values)):
        current.next = ListNode(values[i])
        current = current.next
    return head

def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


class Solution206:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        pre = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

sol = Solution206()
test_cases = [
    ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
    ([], []),
    ([1, 2], [2, 1]),
    ([1], [1]),
]
for i, (input_list, expected_output) in enumerate(test_cases):
    head = create_linked_list(input_list)
    reversed_head = sol.reverseList(head)
    actual_output = linked_list_to_list(reversed_head)
    assert actual_output == expected_output, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```


## 二叉树

### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum)

* 方法: 递归, DFS
* 时间复杂度: O(N) -- 每个元素遍历一遍.
* 空间复杂度: O(H) -- 需要引入调用栈, 栈的大小取决于最深的路径有多长.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def create_tree_from_list(values):
    from collections import deque
    if not values:
        return None
    root = TreeNode(values[0])
    q = deque([root])
    i = 1

    while q and i < len(values):
        current_node = q.popleft()
        if values[i]:
            current_node.left = TreeNode(values[i])
            q.append(current_node.left)
        i += 1

        if i >= len(values):
            break

        if values[i]:
            current_node.right = TreeNode(values[i])
            q.append(current_node.right)
        i += 1
    return root

class Solution124:
    def maxPathSum(self, root: TreeNode) -> int:
        self.global_gain = -float("inf")  # 无法确定最大和是正数还是负数, 所以从负无穷开始更新全局结果.
        def compute_sum(root: TreeNode):
            if not root:
                return 0
            left_gain = max(0, compute_sum(root.left))
            right_gain = max(0, compute_sum(root.right))
            self.global_gain = max(root.val + left_gain + right_gain, self.global_gain)
            return root.val + max(left_gain, right_gain)
        compute_sum(root)
        return self.global_gain
    
sol = Solution124()
test_cases = [
    ([1, 2, 3], 6),
    ([-10, 9, 20, None, None, 15, 7], 42)
]
for i, (root, res) in enumerate(test_cases):
    assert sol.maxPathSum(create_tree_from_list(root)) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```




### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree)
* **方法**: 递归 + 后序遍历 -- 该方法, 即 lowestCommonAncestor 的准确定义是, 在 root 为根的子树中对 p 和 q 进行查找, 并返回该子树中, 能覆盖住 p 或 q（或两者）的最近节点.
* **时间复杂度**: O(n) -- 最坏情况下, 如 p 和 q 都在最底部或树退化为列表, 那么需要遍历树的每一个节点.
* **空间复杂度**: O(log n) -- 这题是普通二叉树, 需要后序遍历汇总信息, 此时存在隐形的系统栈空间, 由栈深度决定空间复杂度, 平均为 O(log n).




```python
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def create_tree_from_list(values: List[int]):
    from collections import deque
    root = TreeNode(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        current_node = q.popleft()
        if values[i]:
            current_node.left = TreeNode(values[i])
            q.append(current_node.left)
        i += 1
        if i >= len(values):
            break
        if values[i]:
            current_node.right = TreeNode(values[i])
            q.append(current_node.right)
        i += 1
    return root

nodes_map = {}

def build_map(node: TreeNode):
    if not node:
        return
    nodes_map[node.val] = node
    build_map(node.left)
    build_map(node.right)

class Solution236:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if left and not right:
            return left
        return right
    
sol = Solution236()
test_cases = [
    ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 1, 3),
    ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 4, 5),
    ([1, 2], 1, 2, 1)
]
for i, (tree, p, q, res) in enumerate(test_cases):
    root = create_tree_from_list(tree)
    build_map(root)
    assert sol.lowestCommonAncestor(root, nodes_map.get(p), nodes_map.get(q)) == nodes_map.get(res), f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```


### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree)

```python
class Solution226:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left, root.right = right, left
        return root


```



## 图论

### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands)

* 方法: 
    1. DFS -- 遍历整个二维网格. 一旦发现一个值为 '1' 的未访问过的陆地格子, 就将其视为一个新岛屿的起点, 并将岛屿计数器加一. 随后, 从该点开始进行递归的深度优先搜索, 将所有与之直接或间接相连的 '1' 格子全部标记为已访问状态 (如 '2'), 从而确保同一个岛屿不会被重复计数. 
    2. BFS -- 遍历网格寻找 '1'. 发现后, 计数器加一, 并初始化一个队列, 将当前格子的坐标入队. 接着, 启动一个循环, 只要队列不为空, 就取出队首元素, 并将其所有相邻且未被访问的陆地邻居标记后加入队尾, 逐层将整个岛屿标记完毕.
* 时间复杂度:  O(M * N) (DFS & BFS) -- DFS 与 BFS 均为 O(M * N) -- 两种方法的核心都是对网格进行一次完整的遍历. 每个格子 (i, j) 都被主循环检查一次. 如果它是陆地, 那么它会被相应的搜索算法（DFS的递归或BFS的队列操作）处理一次. 由于每个格子在被处理后都会被立刻标记, 保证了它不会被再次处理. 因此, 整体时间复杂度与网格的格子总数 M * N 呈线性关系.
* 空间复杂度: 
    1. DFS (递归实现): O(M * N) -- 空间开销主要源于递归调用栈的深度. 在最坏的情况下, 整个网格可以被一个单一的, 蜿蜒曲折的蛇形岛屿填满. 此时, DFS的递归调用链会延伸至几乎所有格子, 导致调用栈的深度接近 M * N.
    2. BFS (队列实现): O(min(M, N)) -- 将BFS的探索过程想象成一个从起点开始, 逐层向外扩散的波纹. 队列存储的就是这个波纹在某一时刻的“波前” (Frontier), 即所有待探索的边界节点. 为了分析其最坏情况, 我们假设一个M行N列的网格完全被单一岛屿覆盖. 当BFS从一个角落(例如左上角)开始时, 这个“波前”会近似于一条对角线, 向网格的对角(右下角)推进. 在推进过程中, 队列的长度(即“波前”的节点数)会不断增加. 当这个“波前”的长度增长到第一次同时触及网格的两条相对边界时, 其长度达到峰值. 这个峰值长度受限于网格较短的那条边. 例如, 在一个M行N列且M小于N的矩形中, “波前”会在其长度增长到M左右时触及上下边界, 此后无论如何向右推进, 其长度都无法再超过M. 因此, 队列的最大长度与网格的短边长度成正比, 空间复杂度为 O(min(M, N)). 

```python
from typing import List
from collections import deque
import copy

class Solution200:
    def numIslandsDFS(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '2'
            dfs(i - 1, j)
            dfs(i, j - 1)
            dfs(i + 1, j)
            dfs(i, j + 1)
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    ans += 1
        return ans

    def numIslandsBFS(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    q = deque([(i, j)])
                    grid[i][j] = '2'
                    while q:
                        row, col = q.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            new_row, new_col = row + dr, col + dc
                            if 0 <= new_row < m and 0 <= new_col < n and grid[new_row][new_col] == '1':
                                grid[new_row][new_col] = '2'
                                q.append((new_row, new_col))
        return ans

sol = Solution200()
test_cases = [
    ([["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]], 1),
    ([["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]], 3)
]
for i, (grid, res) in enumerate(test_cases):
    grid_copy = copy.deepcopy(grid)
    assert sol.numIslandsDFS(grid) == res and sol.numIslandsBFS(grid_copy) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```





## 堆


### [215. 数组中的第 K 个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array)

* 方法: 快速选择 -- 题解的本质是要找到排序后的数组中索引为 n - k 的数, 但我们不需要真正排序! 每次随机选取一个索引, 叫 pivot, nums 里 pivot 左侧的数都小于等于 pivot 指向的数, 右边的数都大于等于 pivot 指向的数. 如果 pivot 等于 n - k, 那我们就找到了答案, 若不等, 我们就把 pivot 不包含 n - k 索引的那一侧的数字全部抛弃, 在另一侧继续按类似原理找到下一个 pivot, 每次丢掉大约一半的数据, 所以最后被分类过的数据量在最坏情况下就是 2N (1 + N / 2 + N / 4 + ...).
* 时间复杂度: O(N) -- 根据上面讲解, 时间复杂度为 O(2N), 即 O(N).
* 空间复杂度: O(1) -- 
* 备注: "如果这个数组非常大, 内存存不下怎么办" 或者 "如果数据是源源不断流进来的 (直播间弹幕), 你怎么实时找到第 K 大?" 这时候, 快速选择 (Quick Select) 就不行了. 因为快速选择必须把所有数据一次性加载到内存里, 还需要随机访问 (Random Access). 而堆可以解决这些问题: 它只需要占用 O(K) 的内存 (比如找前 100 大, 只存 100 个数). 数据来一个处理一个, 处理完就扔, 不需要保存历史数据. 所以, 在工业界 (大数据处理, 实时计算) 中, Top K 问题几乎都是用堆或者类似的 Sketch 算法解决的.

```python
from random import randint
from typing import List


class Solution215:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        target_index = n - k
        left, right = 0, n - 1

        while True:
            pivot_index = self.partition(nums, left, right)

            if pivot_index == target_index:
                return nums[pivot_index]
            elif pivot_index > target_index:
                right = pivot_index - 1
            else:
                left = pivot_index + 1

    def partition(self, nums, left, right):
        # 随机选择 pivot 并移到最左边，避免最坏情况.
        random_pivot = randint(left, right)
        nums[left], nums[random_pivot] = nums[random_pivot], nums[left]

        pivot = nums[left]
        lt = left + 1
        rt = right

        while True:
            # 取等号是因为要确保重合数据点也被严格归类.
            while lt <= rt and nums[lt] < pivot:
                lt += 1
            while lt <= rt and nums[rt] > pivot:
                rt -= 1

            if lt >= rt:
                # 只存在相等 (都指向了 nums[pivotal] 相等的数) 和错位 (rt 在 lt 左边一位) 两种情形.
                break

            # 交换不符合顺序的元素
            nums[lt], nums[rt] = nums[rt], nums[lt]
            lt += 1
            rt -= 1

        # 因为 pivot 此刻在 "小于区", 所以必须和 "小于区" 的元素做交换
        nums[left], nums[rt] = nums[rt], nums[left]
        return rt

sol = Solution215()
test_cases = [
    (2, [3,2,1,5,6,4], 5),
    (4, [3,2,3,1,2,4,5,5,6], 4)
]
for i, (k, nums, res) in enumerate(test_cases):
    assert sol.findKthLargest(nums, k) == res, f"test cases {i + 1} failed."
print("\n All test cases passed successfully!")
```