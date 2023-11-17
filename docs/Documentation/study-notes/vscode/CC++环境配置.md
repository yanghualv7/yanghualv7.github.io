# C/C++环境配置

## 前言

参考链接：**bilibili up主 _重装小兔19C**[完美配置vscode运行c/c++环境，2022最新教程](https://www.bilibili.com/video/BV1ge4y1J72t/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=7cbfb227bb3d5386a067808fd725b17b)

环境：Windows 10 ，vscode， mingw64

在学会安装配置前，首先要学会卸载！！！！



## 一. 完全卸载vscode

### 1.卸载vscode

$$
设置 \rightarrow 应用 \rightarrow Microsoft\ Visual\ Studio\ Code (User)
$$



![image-20231117133733277](CC++环境配置.assets/image-20231117133733277.png)



### 2.删除 `.vscode` 配置文件

​	该路径下保存的是`Visual Studio Code` 的项目特定配置文件的目录。其中可以包含一些配置文件，例如 `settings.json` 和 `launch.json` 用于配置工作空间的设置和调试器的启动配置，还包括了 `tasks.json` 用于配置任务运行器，路径如下。
$$
C:\Users\yhl
$$




![image-20231117134048505](CC++环境配置.assets/image-20231117134048505.png)



### 3.删除`Code` 文件夹

​	`Visual Studio Code` 的用户配置和扩展文件存储的默认路径。在这个目录下，你会找到用户设置、已安装的扩展、代码片段和其他用户相关的配置文件。这个文件夹存储了你在使用 Visual Studio Code 过程中产生的一些个性化设置和数据,路径如下。
$$
C:\Users\yhl\AppData\Roaming
$$

**注意** ：AppData文件夹找不到时，请将 **隐藏项目** 勾选

![image-20231117134847580](CC++环境配置.assets/image-20231117134847580.png)



完成上述操作后，vscode已经在电脑上彻底删除了。



## 二. 安装vscode

​	[vscode下载地址](https://code.visualstudio.com/)

本文不过多赘述，请自行下载，并安装至适当位置。



## 三.安装mingw6



### 1.下载mingw6

​	[mingw6下载地址](https://nuwen.net/mingw.html)

​	**mingw6** 是指**“Minimalist GNU for Windows 6”**，是一个用于在 **Windows** 操作系统上进行开发的开源工具集。它提供了一组 **GNU** 工具，例如 **GCC** 编译器、**GNU Make** 等，使开发者能够在 **Windows** 平台上进行类 Unix 环境下的开发工作。这些工具包括了 **C、C++、Objective-C、Fortran** 等语言的编译器和相关工具，使得开发者可以在 **Windows** 上进行跨平台的开发工作。mingw6 是在 mingw-w64 的基础上进行了一些改进，提供了更多的功能和性能改进。



![image-20231117135725166](CC++环境配置.assets/image-20231117135725166.png)



### 2.安装mingw6

 	请注意，路径不要存在空格和中文

![image-20231117140038739](CC++环境配置.assets/image-20231117140038739.png)



### 3.配置mingw6环境变量

打开mingw6安装的路径，进入 **bin** 目录，复制该路径，以下是作者安装路径，读者根据自己的路径进行更换。
$$
D:\MinGW\bin
$$


![image-20231117140528964](CC++环境配置.assets/image-20231117140528964.png)



### 4.进入系统环境变量

![image-20231117140735123](CC++环境配置.assets/image-20231117140735123.png)

依次点击：

作者一般选择系统环境变量，读者可根据自己的需求选择
$$
高级 \rightarrow 环境变量 \rightarrow 用户环境变量/系统环境变量（前者为当前用户，后者为所有用户）\rightarrow Path \rightarrow 新建
$$
将上一步复制的路径填入，然后点击 **3次确定** 



### 5.查看环境变量是否设置成功

 `win + r`  打开cmd，输入以下命令

```cmd
gcc -v

g++ -v
```



结果展示

![image-20231117142112089](CC++环境配置.assets/image-20231117142112089.png)



## 四.配置vscode

### 1.安装扩展

```json
coderunner  C/C++ 

可选插件：Chinese (Simplified) (简体中文) Language Pack 
安装后会提示重启生效

```



![image-20231117142448475](CC++环境配置.assets/image-20231117142448475.png)



![image-20231117142613026](CC++环境配置.assets/image-20231117142613026.png)



### 2.配置Code Runner扩展

​	点击**coderunner**扩展，按图示指引操作

![image-20231117143117384](CC++环境配置.assets/image-20231117143117384.png)



​	找到run in terminal 和save file before run（在运行之前保存），将其勾选



### 3.配置C/C++扩展

- 找到compiler path填入以下内容

**注意** : D:\\MinGW\\bin\\g++.exe 修改为你电脑上的实际路径

```json
{
    "code-runner.runInTerminal": true,
    "code-runner.saveFileBeforeRun": true,
    "C_Cpp.default.compilerPath": "D:\\MinGW\\bin\\g++.exe"
}
```



![image-20231117144018161](CC++环境配置.assets/image-20231117144018161.png)



-  选择Cpp Standard和C Standard版本，选择C11/C14 均可

![image-20231117144414558](CC++环境配置.assets/image-20231117144414558.png)



- 添加 Include Path

![image-20231117144803004](CC++环境配置.assets/image-20231117144803004.png)



- 选择Intelli Sense Mode， 选择gcc-x64



检查完毕后关闭此扩展



## 五.测试代码


$$
文件 \rightarrow 打开文件夹 \rightarrow 这里选择一个你存放代码的文件夹
$$
![image-20231117145718387](CC++环境配置.assets/image-20231117145718387.png)

![image-20231117145908775](CC++环境配置.assets/image-20231117145908775.png)



![image-20231117150032954](CC++环境配置.assets/image-20231117150032954.png)

![image-20231117150112369](CC++环境配置.assets/image-20231117150112369.png)



### 1.更换C/C++扩展版本

​	作者使用 `1.15.4`, 读者可根据具体需求选择版本。

![image-20231117151834725](CC++环境配置.assets/image-20231117151834725.png)



### 2.使用cmd终端

​	按图示操作，将 `lanuch.json`  内的 `externalConsole` 值改为 `true`

**注意** ：尽量选择 **g++** ，选择 **gcc** 可能会出现问题。

![image-20231117152216487](CC++环境配置.assets/image-20231117152216487.png)



### 3.效果展示

![image-20231117152531139](CC++环境配置.assets/image-20231117152531139.png)



**注意**： 当窗口出现闪退时，仅需在 **return 0** 前加入 s**ystem("pause");** 即可。

例如下方示例代码

```c++
int main()
{
	char binary[100];
	int n;
	printf("输入二进制数位数:\n");
	scanf_s("%d", &n, sizeof(n));
	getchar(); // 清空输入缓冲区
	printf("输入二进制数:\n");
	scanf_s("%[^\n]", binary, sizeof(binary));
	printf("带符号整数:%d\n", intToValue(binary, n));
	printf("无符号整数:%u\n", unsignedToValue(binary, n));
	system("pause");
	return 0;

}
```



## 六.结语

​	在完成以上配置后，你已经成功搭建了C/C++开发环境，并可以在vscode中进行代码编写、编译和运行。通过这样的环境配置，你可以更便捷地进行C/C++语言的学习和开发工作。祝你在编程学习和项目开发中取得成功！