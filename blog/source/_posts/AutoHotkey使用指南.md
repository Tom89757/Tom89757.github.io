---
title: AutoHotkey使用指南
date: 2023-05-31 15:30:49
categories:
- 开发工具
tags:
- AutoHotkey 
---
本文记录一下所使用的AutoHotkey脚本：
<!--more-->
### 快捷键`Ctrl+shift+g`使用google搜索剪贴板 (clipboard) 的内容
```bash
^+g::
    {
        Send, ^c
        Sleep 50
        Run, https://www.google.com/search?q=%clipboard%
        Return
    }
```
> 参考资料：
> 1. [10 Cool AutoHotkey Scripts (And How to Make Your Own!)](https://www.makeuseof.com/tag/10-cool-autohotkey-scripts-make/)

### 双击选中单词并将其复制到剪贴板 (clipboard)，使用golden dict搜索
```bash
~LButton::

  Loop {
    LButtonDown := GetKeyState("LButton","P") 
    If (!LButtonDown)
      Break
  }

WaitTime:=DllCall("GetDoubleClickTime")/4000
KeyWait, LButton, D T%WaitTime%
If errorlevel=0
   GoSub, Routine
Return



Routine:
{

  ifwinactive ahk_class CabinetWClass
  {
    return
  }

  clipboard = 
  send ,^c
  ClipWait,1

  StringLen, cliplen, clipboard
  if cliplen > 20
  { 
    ;避免不是英文單字的東西送到GoldenDict去翻譯。
    return
  }

  if cliplen < 2
  {   
    ;避免不是英文單字的東西送到GoldenDict去翻譯。
    return
  }


; send,{Ctrl down}cc{Ctrl up} 可用這行，也可用下行

run D:\Tools\GoldenDict\GoldenDict.exe  %clipboard%

}

return
```
> 参考资料：
> 1. [goldendict词典如何实现高亮取词（双击取词）？ - 知乎](https://www.zhihu.com/question/291320101)


### 在当前目录使用`Ctrl+shift+t`新建text文件

> 参考资料：
> 1. [Windows新建文件快捷键使用AutoHotKey脚本_✿三日月的博客-CSDN博客](https://blog.csdn.net/qq_44119557/article/details/123284048)
> 2. [Script to create and open a new text file? - AutoHotkey Community](https://www.autohotkey.com/boards/viewtopic.php?t=64289)：打开nodepad
> 3. [AutoHotKey: Create a new file with Alt+N keyboard shortcut in Windows Explorer · GitHub](https://gist.github.com/aborruso/8867d871bbb421495476b22f040f0ee2)：完美满足 