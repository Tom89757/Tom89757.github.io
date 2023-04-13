---
title: ACM格式输入2
date: 2023-04-13 17:00:32
categories:
- 资料 
tags:
- ACM
---

本文转载自 [ACM格式输入（二） | DFSgwb](https://dfsgwb.github.io/2023/04/13/ACM%E6%A0%BC%E5%BC%8F%E8%BE%93%E5%85%A52/)
<!--more-->

# c++常用的输入输出方法
## 案例
一维数组：
>输入包含一个整数n代表数组长度。
>接下来包含n个整数，代表数组中的元素
>3
>1 2 3
```cpp
int n;
scanf("%d",&n); // 读入3，说明数组的大小是3
vector<int> nums(n); // 创建大小为3的vector<int>
for(int i = 0; i < n; i++) {
	cin >> nums[i];
}

// 验证是否读入成功
for(int i = 0; i < nums.size(); i++) {
	cout << nums[i] << " ";
}
cout << endl;

```
若是不限定输入数据的大小
```cpp
vector<int> nums;
int num;
while(cin >> num) {
	nums.push_back(num);
	// 读到换行符，终止循环
	if(getchar() == '\n') {
		break;
	}
}
// 验证是否读入成功
for(int i = 0; i < nums.size(); i++) {
	cout << nums[i] << " ";
}
cout << endl;
```

### 二维数组
例如
>输出N行，每行M个空格分隔的整数。每个整数表示该位置距离最近的水域的距离。
>4 4  
>0110  
>1111  
>1111  
>0110

```cpp
int n,m;
int res[n][m];
//vector<vector<int>>res(n,vector<int>(n));
scanf("%d%d",&n,&m);
for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
        scanf("%d",&res[i][j]);
    }
}
// 验证是否读入成功
for(int i = 0; i < m; i++) {
	for(int j = 0; j < n; j++) {
		cout << matrix[i][j] << " ";
	}
	cout << endl;
```