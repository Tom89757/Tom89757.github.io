---
title: Android Studio配置
date: 2022-07-27 14:00:44
categories:
- 开发工具
tags:
- Android Studio
---

本文记录一下在配置Android Studio中一些tips：

<!--more-->

### 通过配置环境变量将SDK和Emulator放在D盘

- 配置`ANDROID_SDK_ROOT`环境变量，对应目录下存放安装的`SDK`：

  ![image-20220730191707531](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730191707531.png)

  ![image-20220730191849901](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730191849901.png)

  

- 配置`ANDROID_SDK_HOME`环境变量，对应目录下的`.android\avd`目录中存放安装的安卓模拟器。

  ![image-20220730191737331](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730191737331.png)

  ![image-20220730191916435](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730191916435.png)

  注意将原本C盘中的`.android`文件夹复制到该目录处时，需要更改`avd`下面对应模拟器的`.ini`文件中的路径：

![image-20220730190220294](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730190220294.png)

> 参考资料：
>
> 1. [环境变量](https://developer.android.com/studio/command-line/variables?hl=zh-cn)
> 2. [Android Studio 配置模拟器AVD存放路径（默认在c盘，解决c盘空间不够问题）](https://blog.51cto.com/u_15307523/3133953)

### 在Android Studio项目中导入已有安卓模拟器

直接导入对应`avd`目录下的`.ini`文件即可。

### Android Studio各种报错

1.出现报错：`This view is not constrained vertically: at runtime it will jump to the top unless you add a vertical constraint`。出现场景如下：

![image-20220730190618272](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730190618272.png)

此时可以通过点击右上方的魔法工具完整自动补全解决报错：

![image-20220730190838880](https://raw.githubusercontent.com/Tom89757/ImageHost/main/hexo/image-20220730190838880.png)

补全后代码如下：

```xml
<Button
        android:id="@+id/button1"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:text="Button 1"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />
```

> 参考资料：
>
> 1. [This view is not constrained vertically. At runtime it will jump to the left unless you add a vertical constraint](https://stackoverflow.com/questions/37859613/this-view-is-not-constrained-vertically-at-runtime-it-will-jump-to-the-left-unl)

### Android Studio配置国内镜像

### Android Studio中用ViewBinding替代kotlin-android-extension插件

在Activity中使用视图绑定的流程如下：

```kotlin
class FirstActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var binding = FirstLayoutBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        // val button1: Button = findViewById(R.id.button1)
        // Toast.makeText要求传入三个参数，第一个参数为一个Context对象，Activity本身就是一个Context对象，因此传入this即可；第二个参数为显示的文本；第三个参数为显示时长
        binding.button1.setOnClickListener{
            Toast.makeText(this, "You clicked Button 1", Toast.LENGTH_LONG).show()
        }
    }
}
```

其与以下代码功能等价，但在页面中控件较多时更为高效：

```kotlin
class FirstActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.first_layout)
        val button1: Button = findViewById(R.id.button1)
        // Toast.makeText要求传入三个参数，第一个参数为一个Context对象，Activity本身就是一个Context对象，因此传入this即可；第二个参数为显示的文本；第三个参数为显示时长
        button1.setOnClickListener{
            Toast.makeText(this, "You clicked Button 1", Toast.LENGTH_LONG).show()
        }
    }
}
```

> 参考资料：
>
> 1. [kotlin-android-extensions插件也被废弃了？扶我起来](https://blog.csdn.net/guolin_blog/article/details/113089706)
> 2. [视图绑定](https://developer.android.com/topic/libraries/view-binding) 官方文档



























