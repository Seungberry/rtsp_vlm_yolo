# RTSP-YOLO-VLM: 基于 YOLOv8-Pose 与 Qwen-VL 的多模态实时视频流行为分析系统

本项目实现了一个高可用、低延迟的边缘端智能视频分析流媒体管线。系统通过 RTSP 协议接收原始视频流，利用 YOLOv8-Pose实时捕获并绘制人体骨骼关键点，同时通过多线程异步锁机制动态触发多模态视觉大模型（Qwen3-VL-Flash）对画面中人物的动作、姿态及深层意图进行行为描述，最后通过 FFmpeg 管道将复合可视化的彩色流逆向推至媒体服务器。

## 核心特性

* 实时高性能骨骼关键点提取：基于 `yolov8n-pose` 模型，在边缘端以极低延迟（Zero-Latency调优）自动提取并绘制人体的边界框、关键点及骨骼连线。
* 非阻塞多线程 VLM 异步调用：设计了行为触发器锁机制（`vlm_is_running` 状态机），在画面中检测到人且达到设定时间间隔（如 5s）时，异步抽取当前帧进行 Base64 编码并调用多模态大模型，完全不阻塞、不降低实时推流的帧率。
* 低级别管道推流管线：绕过高层包装，直接使用 Python 的 `subprocess.Popen` 构建底层 `FFmpeg` 进程管道，将 BGR24 格式的高清可视化流以 `libx264` 编码实时逆向推送至 RTSP 转发服务器（如 EasyDarwin）。
* 零延迟推流调优：预设了 `-preset ultrafast` 和 `-tune zerolatency` 等工业级流媒体参数，确保在有限带宽和算力下实现画面超低延时同步。

## 系统架构与数据流流向
<img width="975" height="529" alt="image" src="https://github.com/user-attachments/assets/314ef390-6de0-414b-8254-5055c7b1f2bb" />


## 技术栈与依赖

* 核心框架：`Ultralytics (YOLOv8)`
* 流媒体/图像处理：`OpenCV-Python`, `FFmpeg`
* 大模型生态：`LangChain-OpenAI` (用于标准封装接入阿里 DashScope 多模态服务)
* 硬件加速：`CUDA` (强烈建议用于 YOLO 实时推理与渲染)

## 核心文件结构

* `yolo.py`：基础推流管线实现。展示了如何通过流式 `stream=True` 读取视频，利用 YOLO 快速渲染骨骼线并通过 FFmpeg 稳定推流的核心骨架。
* `rtsp_vlm_yolo.py`：高级生产级应用。在推流骨架基础上，加入了多线程异步大模型决策层（`qwen3-vl-flash`），实现了“实时推流+高层语义分析”的双轨并行。

## 快速开始

### 1. 启动流媒体服务器

本系统依赖 RTSP 转发服务器，推荐在本地启动 [EasyDarwin](https://www.google.com/search?q=http://www.easydarwin.org/)：

* 控制台访问：`http://localhost:10008`
* 推流路径配置：输入路径 `rtsp://127.0.0.1:25544/input`，输出路径 `rtsp://127.0.0.1:25544/output`

### 2. 环境配置

```bash
pip install opencv-python ultralytics langchain-openai numpy

```

*请确保系统环境中 `ffmpeg` 命令处于环境变量中可直接调用。*

### 3. 配置密钥并运行

在 `rtsp_vlm_yolo.py` 中配置大模型服务密钥：

```python
chat_model = ChatOpenAI(
    openai_api_key="your_dashscope_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    model="qwen3-vl-flash" # 可根据实际模型权限进行调整
)

```

执行核心脚本：

```bash
python rtsp_vlm_yolo.py

```
<img width="1426" height="872" alt="屏幕截图 2026-05-25 094727" src="https://github.com/user-attachments/assets/00cc2120-e331-4cb2-ac24-1a567d6c212f" />
