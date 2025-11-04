#include <iostream>
#include <vector>
#include <queue>
#include <algorithm> // 用于 std::max
#include <limits>    // 用于 std::numeric_limits
#include <cassert>   // 用于 assert
// g++ 124.cpp -o solution -std=c++11
// 定义一个代表空节点的常量
const int NULL_NODE = std::numeric_limits<int>::min();

// C++版本的二叉树节点定义
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// C++版本的从列表创建二叉树的函数
// 注意：这里的列表是 std::vector<int>
TreeNode* create_tree_from_list(const std::vector<int>& values) {
    if (values.empty() || values[0] == NULL_NODE) {
        return nullptr;
    }

    TreeNode* root = new TreeNode(values[0]);
    std::queue<TreeNode*> q;
    q.push(root);
    int i = 1;

    while (!q.empty() && i < values.size()) {
        TreeNode* current_node = q.front();
        q.pop();

        // 处理左子节点
        if (values[i] != NULL_NODE) {
            current_node->left = new TreeNode(values[i]);
            q.push(current_node->left);
        }
        i++;

        if (i >= values.size()) {
            break;
        }

        // 处理右子节点
        if (values[i] != NULL_NODE) {
            current_node->right = new TreeNode(values[i]);
            q.push(current_node->right);
        }
        i++;
    }
    return root;
}

// 解法类的C++实现
class Solution124 {
private:
    int global_gain;

    // 递归辅助函数，计算从当前节点出发向下的最大路径和
    int compute_sum(TreeNode* root) {
        if (!root) {
            return 0;
        }

        // 递归计算左右子树的贡献值
        // 如果子树的贡献值为负，则不选择它，计为0
        int left_gain = std::max(0, compute_sum(root->left));
        int right_gain = std::max(0, compute_sum(root->right));

        // 更新全局最大路径和
        // 这个路径可以穿过当前节点，连接其左右子树
        int current_path_sum = root->val + left_gain + right_gain;
        this->global_gain = std::max(this->global_gain, current_path_sum);

        // 返回从当前节点出发，只能向“上”延伸的路径的最大和
        return root->val + std::max(left_gain, right_gain);
    }

public:
    int maxPathSum(TreeNode* root) {
        // 初始化全局最大值为负无穷大
        this->global_gain = std::numeric_limits<int>::min();
        compute_sum(root);
        return this->global_gain;
    }
};

// 主函数，用于测试
int main() {
    Solution124 sol;

    // === 测试用例 1 ===
    // 在C++中，我们用之前定义的 NULL_NODE 来代表 None
    std::vector<int> tree1_values = {1, 2, 3};
    TreeNode* root1 = create_tree_from_list(tree1_values);
    int expected1 = 6;
    assert(sol.maxPathSum(root1) == expected1);
    std::cout << "Test case 1 passed." << std::endl;


    // === 测试用例 2 ===
    std::vector<int> tree2_values = {-10, 9, 20, NULL_NODE, NULL_NODE, 15, 7};
    TreeNode* root2 = create_tree_from_list(tree2_values);
    int expected2 = 42;
    assert(sol.maxPathSum(root2) == expected2);
    std::cout << "Test case 2 passed." << std::endl;


    std::cout << "\nAll test cases passed successfully!" << std::endl;

    // 在C++中，手动创建的内存需要手动释放，但在本示例中为简化而省略
    // 实际项目中需要写一个函数来释放树的内存
    return 0;
}