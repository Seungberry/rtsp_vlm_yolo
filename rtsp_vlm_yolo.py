import cv2
import subprocess
import numpy as np
import threading
import time
import base64
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 1. 配置区域 
RTSP_INPUT_URL = "rtsp://127.0.0.1:25544/input"
RTSP_OUTPUT_URL = "rtsp://127.0.0.1:25544/output"

# 视觉大模型配置
chat_model = ChatOpenAI(
    openai_api_key="sk-5aea0f9329ae48c791a9e18c6e61dccf",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    model="qwen3-vl-flash",
    temperature=0.7,
    max_tokens=1024,
)

# 初始化 YOLO 模型
yolo_model = YOLO("yolov8n-pose.pt")

# 全局变量控制 VLM 触发频率
last_vlm_time = 0
VLM_INTERVAL = 5  # 每隔 5 秒最多调用一次大模型，防止 API 频率超限或卡顿
vlm_is_running = False  # 锁：确保同一时间只有一个 VLM 请求在后台运行


# 2. 视觉大模型异步线程
def call_vlm_async(frame_bgr, question="描述一下当前画面中人物的动作和环境。"):
    """在后台线程中将图片转为base64并调用大模型，避免阻塞主推流视频"""
    global vlm_is_running
    try:
        # 将 OpenCV 的 BGR 图像编码为 JPG 内存数据
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_str}"
        
        # 构造多模态消息
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
        ]
        
        print("\n [VLM] 正在分析画面...")
        response = chat_model.invoke(messages)
        print(f"\n [AI 实时分析报告]:\n{response.content}\n")
        
    except Exception as e:
        print(f"\n [VLM] API 调用失败: {str(e)}")
    finally:
        vlm_is_running = False  # 释放锁


# 3. 主程序推流逻辑
def main():
    global last_vlm_time, vlm_is_running

    cap = cv2.VideoCapture(RTSP_INPUT_URL)
    if not cap.isOpened():
        print(f"无法打开输入视频流：{RTSP_INPUT_URL}")
        return

    # 获取视频真实参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 544
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 544
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    print(f"输入视频分辨率: {width}x{height}, FPS: {fps}")

    # FFmpeg 推流命令配置
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',       
        '-s', f'{width}x{height}',  # 动态匹配输入的分辨率
        '-r', str(fps),            
        '-i', '-', 
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-crf', '28', 
        '-threads', '4', 
        '-f', 'rtsp',
        RTSP_OUTPUT_URL
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("实时 YOLO + VLM 融合推流系统已启动...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，退出")
            break

        # 3.1 运行 YOLO 关键点识别
        results = yolo_model(frame, stream=True, verbose=False)
        
        # 3.2 绘制 YOLO 骨骼线
        annotated_frame = frame.copy()
        has_person = False
        
        for r in results:
            annotated_frame = r.plot()
            # 判断画面里有没有检测到人 (只要 boxes 数量大于 0)
            if len(r.boxes) > 0:
                has_person = True

        # 3.3 触发视觉大模型 (条件：画中有人 & 达到时间间隔 & 后台无未完成的请求)
        current_time = time.time()
        if has_person and (current_time - last_vlm_time > VLM_INTERVAL) and not vlm_is_running:
            vlm_is_running = True
            last_vlm_time = current_time
            
            # 传入原图(frame)或者带有骨骼线的图(annotated_frame)。这里推荐原图，大模型看得更准。
            # 启动独立线程，不卡顿主循环
            vlm_thread = threading.Thread(
                target=call_vlm_async, 
                args=(frame.copy(), "描述画面中这个人的动作、姿态以及他在做什么。")
            )
            vlm_thread.daemon = True
            vlm_thread.start()

        # 3.4 实时写入 FFmpeg 的 stdin 保证推流流畅
        try:
            process.stdin.write(annotated_frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg 进程已意外关闭")
            break

    # 4. 清理资源
    cap.release()
    if process.poll() is None:
        process.stdin.close()
        process.wait()
    print("系统安全退出")

if __name__ == "__main__":
    main()