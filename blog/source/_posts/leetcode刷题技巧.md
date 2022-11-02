---
title: leetcode刷题技巧
date: 2022-11-02 11:29:04
categories:
- 笔记
tags:
- leetcode
---

本文记录一下在刷leetcode算法题过程中积累的一些技巧和方法：
<!--more-->
### Java本地调试
- 以`81.搜索旋转排序数组-ii.java`为例，其代码如下：
```java
/*
 * @lc app=leetcode.cn id=81 lang=java
 *
 * [81] 搜索旋转排序数组 II
 */

// @lc code=start
class Solution {
    public boolean search(int[] nums, int target) {
        int first = 0;
        int last = nums.length;
        int mid;
        while (first != last) {
            mid = (first+last)>>>1;
            if (nums[mid] == target)
                return true;
            if (nums[first] < nums[mid]) {
                if (nums[first] <= target && target < nums[mid]) {
                    last = mid;
                } else {
                    first = mid + 1;
                }
            } else if (nums[first] > nums[mid]) {
                if (nums[mid] < target && target <= nums[last - 1]) {
                    first = mid + 1;
                } else {
                    last = mid;
                }
            } else {
                // nums[first]==nums[mid]，由于有重复元素存在，不能判断增减性，first加1
                first++;
            }
        }
        return false;
    }
}
// @lc code=end
```
- 在上述文件夹下新键`Test`类即`Test.java`文件用于测试，并将上述`Solution`类名改为`Solution`避免冲突：
```java
public class Test {
    public static void main(String[] args) {
        S s1 = new S();
        int[] nums = new int[]{1, 0, 1, 1, 1};
        int target = 0;
        System.out.println(s1.search(nums, target));
    }
}
```
- 在下述位置打断点并点击`Debug`按钮进入调试：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221102180444.png)
- 注意需要安装`Extension Pack for Java`以及`Code Runner`等插件，系统里需要安装java jdk。
> 参考资料：
> 1. [免费 Leetcode Debug？教你用VSCode优雅的刷算法题！！！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1qL4y1q7bY)
### C++本地调试
- 同样以`81.cpp`为例，注意进行C++调试时`.cpp`文件和路径名中不要含有中文。`81.cpp`代码如下：
```cpp
/*
 * @lc app=leetcode.cn id=81 lang=cpp
 *
 * [81] 搜索旋转排序数组 II
 */

// @lc code=start
#include <bits/stdc++.h>
using namespace std;

//代码
//分析：允许重复元素，33题中如果 A[mid]>=A[first]，则[first, mid]为递增序列不再成立。
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int first =0;
        int last = nums.size();
        int mid;
        while(first!=last){
            mid = first + (last-first)/2;
            if(nums[mid]==target) return true;
            if(nums[first]<nums[mid]){
                if(nums[first]<=target && target<nums[mid]){
                    last=mid;
                }else{
                    first=mid+1;
                }
            }else if(nums[first]>nums[mid]){
                if(nums[mid]<target && target<=nums[last-1]){
                    first=mid+1;
                }else{
                    last=mid;
                }
            }else{
                //nums[first]==nums[mid]，因为有重复元素的存在，不能判断增减性，first++
                first++;
            }
        }
        return false;
    }
};
// @lc code=end
```
- 直接在上述文件中添加`main`函数作为调试入口：
```cpp
int main(){
    vector<int> nums;
    nums={1, 0, 1, 1, 1};
    int target = 0;
    Solution s;
    cout << s.search(nums, target) << endl;
}
```
- 在下述位置打断点并点击右上角`Debug C/C++ File`进行调试：
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20221102183138.png)
- 注意在调试之前，需要配置好C++环境（即`.vscode`文件夹），并安装`C/C++`和`Code Runner`等插件。
> 参考资料：
> [leetcode刷题本地调试模板（C++） – SAquariusの梦想屋](https://blog.songjiahao.com/archives/362)
> [C++ `vector<int>&nums` 用法总结 ](https://www.jianshu.com/p/2524c34511f3)
