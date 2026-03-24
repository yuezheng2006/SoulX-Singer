SoulX-Singer大模型本地部署保姆级教程
一、环境准备
在开始之前，请确保你的电脑（推荐 Win10/Win11）已经安装了以下“三大件”：
1、Visual Studio 2022
很多 AI 依赖包在安装时需要编译 C++ 代码。请去微软官网下载 VS 2022 社区版，安装时务必勾选【使用 C++ 的桌面开发】。如果没有它，后面安装依赖时可能会报错。
2、git
用于从 GitHub 上拉取最新的源代码。

3、cuda12.8

请先更新你的 NVIDIA 显卡驱动，然后前往 NVIDIA 官网下载并安装 CUDA Toolkit 12.8。

💡 避坑提示： 本次教程强烈建议大家准备一张显存至少12GB 以上的 NVIDIA 显卡，以保证模型能顺利运行。
二、获取源码与独立环境
步骤 1：下载便携版 Python (Portable Python)
去网上下载一个 Python 3.10.x 版本的便携版压缩包。解压到你的硬盘（例如 D:\Python_SoulX）。里面会有一个 python.exe，这就是我们等下要用的独立环境。

步骤 2：下载 SoulX-Singer 源代码
打开电脑的终端（CMD 或 PowerShell），找一个空间充裕的磁盘，输入以下命令把代码拉取下来：

git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer
三、下载模型权重

你需要下载以下 2 个核心模型：

主模型：SoulX-Singer
https://www.modelscope.cn/models/Soul-AILab/SoulX-Singer



预处理模型：SoulX-Singer-Preprocess
https://www.modelscope.cn/models/Soul-AILab/SoulX-Singer-Preprocess
📂 关键步骤：配置文件夹
下载完成后，打开你刚刚克隆的 SoulX-Singer 源代码文件夹。

在根目录下，手动新建一个文件夹，命名为 model。
将刚才下载的两个模型文件夹，完整地拖进这个 model 文件夹中。
你的文件目录看起来应该是这样的：

SoulX-Singer/
 ├── model/  👈 (你新建的)
 │    ├── SoulX-Singer/              (主模型权重)
 │    └── SoulX-Singer-Preprocess/   (预处理模型权重)
 ├── requirements.txt
 └── ...(其他代码文件)
四、安装依赖包

打开终端，一定要使用你刚才下载的便携版 Python 的绝对路径 来执行安装命令

(假设你的便携版路径是 D:\Python_SoulX\python.exe)

D:\Python_SoulX\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 
--index-url https://download.pytorch.org/whl/cu128
D:\Python_SoulX\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
五、核心逻辑源代码

🎧 SoulxSingerProcessor（音频预处理器）
当你上传一段参考音频时，它负责对音频进行切片、降噪，并提取音高（Pitch）、特征向量等信息，最终转化为大模型能看懂的 metadata（元数据）。
🎤 SoulxSingerWrapper（核心演唱引擎）
它接收 Processor 传来的 metadata，并调用我们放在 model 文件夹里的主模型进行深度推理。它能完美克隆参考音频的音色，并结合你的指令生成全新的歌曲音频。
🖥️ WebUI（可视化交互界面）
将复杂的 Processor 和 Wrapper 逻辑隐藏在后台，利用 Gradio 框架在网页上画出上传框、按钮和播放器。
五、启动WebUI

在终端中，使用你的便携版 Python 运行项目的 WebUI 启动脚本（例如 webui.py，具体以项目提供的入口文件名为准）：

D:\Python_SoulX\python.exe webui.py
当你在控制台看到如下提示时，说明大模型已经成功在你本地苏醒：

Running on local URL:  http://127.0.0.1:7860/
操作指南：

打开你电脑上的任意浏览器（推荐 Chrome 或 Edge）。
在地址栏输入：http://127.0.0.1:7860/ 并回车。
熟悉的图形化界面出现了！上传你的参考人声，输入曲谱/歌词，点击**“生成”**。
稍等片刻（生成速度取决于你的显卡性能），一首由你的专属 AI 歌手演唱的歌曲就诞生啦！
💡 常见问题 Q&A
Q：运行到一半提示 CUDA out of memory 怎么办？
A：这是爆显存了（OOM）。建议在界面中缩短单次生成的音频长度，或者在代码中调低 Batch Size。
Q：为什么第一次生成特别慢？

A：第一次生成时，系统需要将几个 G 的模型权重加载到显卡的显存中，后续连续生成就会快很多。



