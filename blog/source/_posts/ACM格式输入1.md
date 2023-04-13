---
title: ACM格式输入1
date: 2023-04-13 17:00:21
categories:
- 资料 
tags:
- ACM 
---
本文转载自 [ACM格式输入（一） | DFSgwb](https://dfsgwb.github.io/2023/04/13/ACM%E6%A0%BC%E5%BC%8F%E8%BE%93%E5%85%A5/)：
<!--more-->
# c++常用的输入输出方法
## 输入
1.cin
>注意1：cin可以连续从键盘读入数据
>注意2：cin以空格、tab、换行符作为分隔符
>注意3：cin从第一个非空格字符开始读取，直到遇到分隔符结束读取

```cpp
// 用法1，读入单数据
int num;
cin >> num;
cout << num << endl;  // 输出读入的整数num

// 用法2，批量读入多个数据
vector<int> nums(5);
for(int i = 0; i < nums.size(); i++) {
	cin >> nums[i];
}
// 输出读入的数组
for(int i = 0; i < nums.size(); i++) {
	cout << nums[i] << " ";
}
```
2.getline()
>当读取的字符串中间存在空格时，cin就不可用了，便可以使用getline()
```cpp
string s;
getline(cin, s);
// 输出读入的字符串
cout << s << endl;
```
3.getchar
```cpp
char ch;
ch = getchar();
// 输出读入的字符
cout << ch << endl;
```
4.scanf()
>使用最多的输入方式
```cpp
//1.输入十进制的数 
int a;
scanf("%d",&a);
scanf("%i",&a);
scanf("%u",&a);
//这三种写法都是可以的 
//2.输入八进制和十六进制数 
int b;
scanf("%o",&b); //八进制 
scanf("%x",&b); //十六进制 
//3.输入实数
int c;
scanf("%f",&c);
scanf("%e",&c);
//这两种写法可以互换 
//4.输入字符和字符串 
char d;
string dd;
scanf("%c",&d); //单个字符 
scanf("%s",&dd); //字符串 
//5.跳过一次输入 
int e;
scanf("%*",&e);
//6.输入长整型数 
int f;
scanf("%ld",&f);
scanf("%lo",&f);
scanf("%lx",&f);
scanf("%l",&f);
//四种写法都可以用 
//7.输入短整型数 
int g;
scanf("%hd",&g);
scanf("%ho",&g);
scanf("%hx",&g);
scanf("%h",&g);
//四种写法都可以用 
//8.输入double型数（小数 
double h;
scanf("%lf",&h);
scanf("%lf",&h);
scanf("%l",&h);
//三种写法都可以用 
//9.域宽的使用 
int i;
scanf("%5d",&i);
//10.特殊占位符 
int j,k;
scanf("%d,%d",&j,&k);
int j,k;
scanf("%d",&j);
printf(","); //cout<<",";
scanf("%d",&k);
```
## 输出
cout，printf随意搭配，就不讲了

下面将一些输入格式
```cpp
#include<iostream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>
#include<limits.h>  //INT_MIN 和 INT_MAX的头文件  

using namespace std;

struct stu {
	string name;
	int num;
};


// 1. 直接输入一个数
int main() {
	int n = 0;
	while (cin >> n) { 
		cout << n << endl;
	}
	return -1;
}

// 2. 直接输入一个字符串
int main() {
	string str;
	while (cin >> str) {
		cout << str << endl;
	}
	return -1;
}

// 3. 只读取一个字符 
int main() {
	char ch;
	//方式1
	while (cin >> ch) {
		cout << ch << endl;
	}
	//方式2： cin.get(ch) 或者 ch = cin.get() 或者 cin.get()
	while (cin.get(ch)) {   
		cout << ch << endl;
	}
	//方式3 ：ch = getchar()  
	while (getchar()) {
		cout << ch << endl;
	}
	return -1;
}


// 3.1给定一个数，表示有多少组数（可能是数字和字符串的组合），然后读取
int main() {
	int n = 0; 
	while (cin >> n) {   //每次读取1 + n 个数，即一个样例有n+1个数 
		vector<int> nums(n);
		for (int i = 0; i < n; i++) {
			cin >> nums[i];
		}
		//处理这组数/字符串
		for (int i = 0; i < n; i++) {
			cout << nums[i] << endl;
		}
	}
	return -1;
}

//3.2 首先给一个数字，表示需读取n个字符，然后顺序读取n个字符
int main() {
	int n = 0;
	while (cin >> n) {  //输入数量
		vector<string> strs;
		for (int i = 0; i < n; i++) {
			string temp;
			cin >> temp;
			strs.push_back(temp);
		}
		//处理这组字符串
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << ' ';
		}
	}
	return 0;
}


//4.未给定数据个数，但是每一行代表一组数据，每个数据之间用空格隔开
//4.1使用getchar() 或者 cin.get() 读取判断是否是换行符，若是的话，则表示该组数（样例）结束了，需进行处理
int main() {
	int ele;
	while (cin >> ele) {
		int sum = ele;
		// getchar()   //读取单个字符
		/*while (cin.get() != '\n') {*/   //判断换行符号
		while (getchar() != '\n') {  //如果不是换行符号的话，读到的是数字后面的空格或者table
			int num;
			cin >> num;
			sum += num;
		}
		cout << sum << endl;
	}
	return 0;
}

//4.2 给定一行字符串，每个字符串用空格间隔，一个样例为一行
int main() {
	string str;
	vector<string> strs;
	while (cin >> str) {
		strs.push_back(str);
		if (getchar() == '\n') {  //控制测试样例
			sort(strs.begin(), strs.end());
			for (auto& str : strs) {
				cout << str << " ";
			}
			cout << endl;
			strs.clear();
		}
	}
	return 0;
}


//4.3 使用getline 读取一整行数字到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。
int main() {
	string input;
	while (getline(cin, input)) {  //读取一行
		stringstream data(input);  //使用字符串流
		int num = 0, sum = 0;
		while (data >> num) {
			sum += num;
		}
		cout << sum << endl;
	}
	return 0;
}


//4.4 使用getline 读取一整行字符串到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。
int main() {
	string words;
	while (getline(cin, words)) {
		stringstream data(words);
		vector<string> strs;
		string str;
		while (data >> str) {
			strs.push_back(str);
		}
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << " ";
		}
	}
}

//4.5 使用getline 读取一整行字符串到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。每个字符中间用','间隔
int main() {
	string line;
	
	//while (cin >> line) {  //因为加了“，”所以可以看出一个字符串读取
	while(getline(cin, line)){
		vector<string> strs;
		stringstream ss(line);
		string str;
		while (getline(ss, str, ',')) {
			strs.push_back(str);
		}
		//
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << " ";
		}
		cout << endl;
	}
	return 0;
}



int main() {
	string str;

	
	//C语言读取字符、数字
	int a;
	char c;
	string s;

	scanf_s("%d", &a);
	scanf("%c", &c);
	scanf("%s", &s);
	printf("%d", a);


	//读取字符
	char ch;
	cin >> ch;
	ch = getchar();
	while (cin.get(ch)) { //获得单个字符
		;
	}
	
	//读取字符串
	cin >> str;  //遇到空白停止
	getline(cin, str);  //读入一行字符串

}
```