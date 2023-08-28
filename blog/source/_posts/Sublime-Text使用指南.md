---
title: Sublime Text使用指南
date: 2023-08-28 13:14:40
categories:
- 环境配置
tags:
- Sublime Text 
---
本文记录一下在 Sublime Text编辑器中一些常用的配置方法。
<!--more-->
### Sublime Text中Ctrl+Shift+P不起作用
解决方案：安装Visual Studio，使用其spyxx.exe工具。顺利找出PicGo占据了`Ctrl+Shift+P`快捷键。
> 参考资料：
> 1. [editor - Command Palette shortcut not working in Sublime Text3 - Stack Overflow](https://stackoverflow.com/questions/46330939/command-palette-shortcut-not-working-in-sublime-text3)
> 2. [Ctrl Shift P not working to show command pallete · Issue #152448 · microsoft/vscode · GitHub](https://github.com/microsoft/vscode/issues/152448)
> 3. [windows - Find out what program is using a hotkey - Super User](https://superuser.com/questions/999106/find-out-what-program-is-using-a-hotkey)
> 4. [delphi - Find out what process registered a global hotkey? (Windows API) - Stack Overflow](https://stackoverflow.com/questions/829007/find-out-what-process-registered-a-global-hotkey-windows-api/43645062#43645062)
> 5. [How do I get Spy++ with Visual Studio 2017? - Stack Overflow](https://stackoverflow.com/questions/43360339/how-do-i-get-spy-with-visual-studio-2017)