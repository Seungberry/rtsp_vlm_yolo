import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# 1. 加载 YOLO 肢体关键点检测模型 (首次运行会自动下载 yolov8n-pose.pt)
# -pose 后缀代表姿态估计模型，n 代表 nano 速度最快
model = YOLO("yolov8n-pose.pt") 

# EasyDarwin的RTSP流地址
rtsp_input_url = "rtsp://127.0.0.1:25544/input"
rtsp_output_url = "rtsp://127.0.0.1:25544/output"

# 打开视频流
cap = cv2.VideoCapture(rtsp_input_url)
if not cap.isOpened():
    print(f"无法打开视频流：{rtsp_input_url}")
    exit()

# 获取输入视频的实际宽高和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

print(f"输入视频分辨率: {width}x{height}, FPS: {fps}")

# 2. FFmpeg 推流配置 (保持 bgr24 格式以传输彩色骨骼线)
ffmpeg_cmd = [
    'ffmpeg',
    '-y', 
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',       
    '-s', f'{544}x{544}', 
    '-r', str(fps),            
    '-i', '-', 
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '28', 
    '-threads', '4', 
    '-f', 'rtsp',
    rtsp_output_url
]

# 启动FFmpeg进程
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 3. 运行 YOLO 肢体关键点识别
    results = model(frame, stream=True, verbose=False)
    
    # 4. 自动绘制骨骼线和关键点
    annotated_frame = frame.copy() # 如果没检测到人，就推流原图
    for r in results:
        # r.plot() 会自动画出人体边界框、彩色关节圆点以及连接关节的骨骼线
        annotated_frame = r.plot() 

        # 【进阶技巧】如果你需要获取具体的坐标数据进行动作判断（比如深蹲、摔倒、举手）：
        # if r.keypoints is not None:
        #     # xyn 是归一化后的坐标，xy 是绝对像素坐标
        #     # shape 为 [检测到的人数, 17, 2]
        #     keypoints_data = r.keypoints.xy.cpu().numpy() 
        #     for person_idx, kpts in enumerate(keypoints_data):
        #         # 比如获取第一个人的右手腕 (第10个点，索引为9) 的 X, Y 坐标
        #         if len(kpts) > 9:
        #             right_wrist_x, right_wrist_y = kpts[9]

    # 5. 将带有人体骨骼线的彩色帧写入 FFmpeg 的 stdin
    try:
        process.stdin.write(annotated_frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg进程已关闭")
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
if process.poll() is None:
    process.stdin.close()
    process.wait()