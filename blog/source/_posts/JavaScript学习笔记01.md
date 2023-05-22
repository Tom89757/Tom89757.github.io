---
title: JavaScript学习笔记01
date: 2023-05-22 17:34:58
categories:
- 前端 
- 笔记
tags:
- JavaScript 
---

本文记录一下学习JavaScript过程中的笔记：
<!--more-->
> JavaScript是一门用来操作网页内容的语言，和HTML/CSS一起使网页变得可交互。
参考资料：[Intro to JavaScript - Google 簡報](https://docs.google.com/presentation/d/1PlcpLE9QAMLi0LtvFbzmGqFka6iTm9NhefaE-b5QwdI/edit#slide=id.g109dff8ec18_0_53)

### 5种基本数据类型
- Boolean：`true`, `false`
- Number：`12, 1.618, -46.7, 0, etc.`
- String：`"hello", "world!", "12", "", etc.`
- Null：`null`，值为`null`
- Undefined：`undefined`，声明但未赋值

### 操作符
`+`, `-`, `*`, `/`。
其中`+`可以用于字符串连接：`"hello" + "world" `。
`===`：三连等号，用于比较reference
`==`：两连等号，强制类型转换后进行比较。

### 语法
- 定义函数：
```javascript
const greatestCommonDiviso= (a, b) => {
	while(b!==0){
		const temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}
```
- 定义variables：可以重新赋值
```javascript
let myBoolean = true;
let myNumber = 12;
let myString = "Hello World!";

myBoolean = false;
myNumber = -5.6;
myString = "";
```
- 定义consts：不可以重新赋值
```javascript
const answerToLife = 42;
```
 - `null vs. undefined`：
```javascript
let firstName; // currentyly, firstName is undefined
firstName = "Albert"; //now assigned to a value
firstName = null; //now is null
```
- `let vs. var`：不要再使用`var`，见参考资料1。

> 参考资料：
> 1. [javascript - What is the difference between "let" and "var"? - Stack Overflow](https://stackoverflow.com/questions/762011/what-is-the-difference-between-let-and-var)

### Output & Alerts
- `console.log(1);`：控制台输出
- `alert("Congratulations!";`：页面弹出prompt。

### Arrays
```javascript
// initialize
let pets = ["flowers", 42, false, "bird"]

// access
console.log(pets[3]); // "bird"

//replace
pets[2] = "hamster"; // ["flowers", 42, hamster, "bird"]

//remove from end
pets.pop();

//add to end
pets.push("rabbit");
```

### Conditonals & Loops
Conditionals：
```javascript
if (hour < 12){
	console.log("Good morning!");
}else if (hour <20){
	console.log("Good afternoon!");
}else {
	console.log("Good night!");
}
```
While loops：
```javascript
let z = 1;
while (z<100) {
	z = z * 2;
	console.log(z);
}
```
For loops
```javascript
const pets = ["cat", "dog", "bird"];

for (let i = 0; i<pets.length; i++){
	console.log(pets[i]);
}

// more pythonic way
for (const animal of pets) {
	console.log(animal);
}
```

### Functions
语法：`(parameters) => { body };`。
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522180406.png)
JavaScript中functions本身也可以作为参数传递，就像其它的variables一样。这意味着我们可以方便地将一个"callback" function传递给另一个function作为参数。
```javascript
const addTwo = x => {
	return x + 1;
};

const modifyArray = (array, callback) => {
	for (let i = 0; i<array.length; i++){
		array[i] = callback(array[i]);
	}
}

let myArray = [5, 10, 15, 20];
modifyArray(myArray, addTwo); // [7, 12, 17, 22]

// Anonymous functions
modifyArray(myArray, x => {
	return x + 2;
});
modifyArray(myArray, x => x + 2); //简写
```
正因为functions作为first-class的特性，JavaScript还内置了有用的array functions便于对array类型变量的操作：
```javascript
let myArray =  [1, 2, 3, 4, 5];
myArray.map(x => x * 3);
myArray.filter(x => x > 0);
```
PS：前面的`push`和`pop`  mutate the target array in-place，`map`和`filter`生成新的array。

### Objects
一个JavaScript `Object`就是a collection of `name:value` pairs。
```javascript
const myCar = {
	make : "Ford", 
	model: "Mustang", 
	year : 2005,
	color : "red"
};

// accessing properties
console.log(myCar.model); // "mustang" 
console.log(myCar["color"]); //"red"

// Object destructuring
// without destructuring
const make = myCar.make;
const model = myCar.model;
// with destructuring
const {make, model} = myCar;
make; // "Ford"
model; // "Mustang"
```

### Equality
`===`：三连等号检查两个variables是否相等
`==`：两连等号在进行强制类型转换后检查两个variables时候相等

### Object references
Object variables为references，它们指向数据实际存储的地方。
```javascript
let person1 = { name: "Bill Gates" };
let person2 = { name: "Bill Gates" };
person1 === person2; //false
```
那么如何拷贝arrays和objects？
```javascript
// assign, 不是copy
let arr = [1, 2, 3];
let copyArr = arr; //将arr的reference赋值为copyArr

// shallow copy, 浅拷贝
// deep copy, 深拷贝
//见参考资料1
```
> 参考资料：
> 1. [Ways to Copy Objects in JavaScript](https://www.javascripttutorial.net/object/3-ways-to-copy-objects-in-javascript/)


### Classes
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522182631.png)
![](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/20230522182637.png)

### setInervals() vs. setTimeout()

> 参考资料：
> 1. [setInterval() global function - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/setInterval)
> 2.[setTimeout() global function - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/setTimeout) 


