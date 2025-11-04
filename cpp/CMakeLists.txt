# 指定需要的 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言 (CXX 代表 C++)
project(MaxPathSumProject CXX)

# 设置 C++ 标准为 C++11 (因为代码中用到了 nullptr 等特性)
# 这等同于之前在 g++ 命令中使用的 -std=c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加一个可执行文件目标
# 它的名字叫 "solution"，由源文件 "solution.cpp" 编译而来
add_executable(solution 124.cpp)