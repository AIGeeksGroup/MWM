"""
机器人导航数据记录脚本
高频记录所有轨迹数据，后续进行筛选处理
"""

import time
import numpy as np
import cv2
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
from queue import Queue

# 导入SDK
from mmk2_sdk import MMK2Robot, RobotMode

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class NavigationDataRecorder:
    """导航数据记录器 - 高频记录所有数据"""
    
    def __init__(
        self,
        robot_ip: str = "192.168.11.200",  # 默认机器人IP
        save_dir: str = "./navigation_raw_data",
        recording_fps: int = 20,  # 记录频率
        camera_resolution: str = "1280,720,30",  # 相机分辨率
        # 可通过官方示例代码run_examples.py查询支持的分辨率
        # 请选择相机RGB输出分辨率:
        # 1. 1280x720 @ 30fps
        # 2. 1280x720 @ 15fps
        # 3. 640x480 @ 30fps
        # 4. 640x480 @ 15fps
        use_async_save: bool = True,  # 是否使用异步保存
        camera_sn: str = "233522073186",  # 默认相机序列号
    ):
        """
        初始化数据记录器
        
        Args:
            robot_ip: 机器人IP地址（默认192.168.11.200）
            save_dir: 数据保存根目录
            recording_fps: 数据记录频率(Hz)
            camera_resolution: 相机分辨率配置
            use_async_save: 是否使用异步保存(推荐开启)
            camera_sn: 相机序列号（默认233522073186）
        """
        self.robot_ip = robot_ip
        self.camera_sn = camera_sn
        self.base_save_dir = Path(save_dir)
        self.recording_fps = recording_fps
        self.camera_resolution = camera_resolution
        self.use_async_save = use_async_save
        
        # 初始化机器人
        logger.info(f"连接机器人: {robot_ip}")
        self.robot = MMK2Robot(ip=robot_ip, mode=RobotMode.REAL)
        
        if not self.robot.is_connected():
            raise ConnectionError("机器人连接失败")
        
        logger.info("✅ 机器人连接成功")
        
        # 生成本次会话的时间戳和目录
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_save_dir / self.session_timestamp
        
        # 创建保存目录
        self._create_directories()
        
        # 打开txt文件用于实时写入
        self.trajectory_txt_path = self.session_dir / f"trajectory_{self.session_timestamp}.txt"
        self.trajectory_file = None  # 将在record()中打开
        self.trajectory_lock = threading.Lock()  # 用于txt文件写入的线程锁
        
        # 数据记录相关
        self.frame_count = 0
        self.trajectory_data = []  # 仍保留用于统计
        self.is_recording = False
        
        # 异步保存队列
        if self.use_async_save:
            self.save_queue = Queue(maxsize=1000)
            self.save_thread = threading.Thread(target=self._async_save_worker, daemon=True)
            self.save_thread_running = False
        
        # 统计信息
        self.start_time = None
        self.dropped_frames = 0
        
    def _create_directories(self):
        """创建数据保存目录结构"""
        # 创建本次会话的目录
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建images子文件夹
        self.image_dir = self.session_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        logger.info(f"数据保存目录: {self.session_dir}")
        logger.info(f"  - 图片目录: {self.image_dir}")
        
    def initialize_camera(self, serial_no: Optional[str] = None):
        """
        初始化头部相机(仅RGB,不启用深度)
        
        Args:
            serial_no: 相机序列号（不提供则使用默认值）
        """
        if serial_no is None:
            serial_no = self.camera_sn
        
        logger.info(f"使用相机SN: {serial_no}")
        
        # 配置头部相机(仅RGB)
        camera_config = {
            "head_camera": {
                "camera_type": "REALSENSE",
                "serial_no": f"'{serial_no}'",
                "rgb_camera.color_profile": self.camera_resolution,
                "enable_depth": "false",
                "align_depth.enable": "false",
            }
        }
        
        logger.info("配置头部相机...")
        self.robot.camera.set_camera_config(camera_config)
        
        logger.info("启动相机流...")
        self.robot.camera.start_stream(["head_camera"])
        
        # 等待相机预热
        time.sleep(2)
        logger.info("✅ 相机初始化完成")
    
    def _async_save_worker(self):
        """异步保存工作线程 - 保存图片并实时写入txt"""
        while self.save_thread_running:
            item = None
            try:
                # 从队列获取数据（1秒超时）
                item = self.save_queue.get(timeout=1.0)
                
                # 检查是否为结束信号
                if item is None:
                    break
                
                frame_id, rgb_image, pose, timestamp = item
                
                # 1. 保存图像 - 使用帧序号命名
                image_path = self.image_dir / f"{frame_id:06d}.jpg"
                cv2.imwrite(str(image_path), rgb_image)
                
                # 2. 实时写入txt文件（线程安全）
                with self.trajectory_lock:
                    if self.trajectory_file and not self.trajectory_file.closed:
                        self.trajectory_file.write(
                            f"{frame_id}\t{pose[0]:.6f}\t{pose[1]:.6f}\t{pose[2]:.6f}\n"
                        )
                        self.trajectory_file.flush()  # 立即刷新到磁盘
                
                # 3. 添加到内存列表用于统计（线程安全）
                with self.trajectory_lock:
                    self.trajectory_data.append([
                        frame_id,
                        float(pose[0]),
                        float(pose[1]),
                        float(pose[2])
                    ])
                
            except Exception as e:
                # 超时（queue.Empty）或其他错误
                if str(type(e).__name__) != 'Empty':  # 不是超时错误才打印
                    logger.error(f"保存错误: {e}")
            finally:
                # 关键：只有成功get()到非None数据才调用task_done()
                if item is not None:
                    self.save_queue.task_done()
    
    def _save_frame_sync(self, frame_id: int, rgb_image: np.ndarray, 
                        pose: np.ndarray, timestamp: float):
        """同步保存帧数据"""
        # 保存图像 - 使用帧序号命名
        image_path = self.image_dir / f"{frame_id:06d}.jpg"
        cv2.imwrite(str(image_path), rgb_image)
        
        # 实时写入txt文件
        if self.trajectory_file and not self.trajectory_file.closed:
            self.trajectory_file.write(
                f"{frame_id}\t{pose[0]:.6f}\t{pose[1]:.6f}\t{pose[2]:.6f}\n"
            )
            self.trajectory_file.flush()
        
        # 添加到内存列表用于统计
        self.trajectory_data.append([
            frame_id,
            float(pose[0]),
            float(pose[1]),
            float(pose[2])
        ])
    
    def _save_frame(self, frame_id: int, rgb_image: np.ndarray, 
                   pose: np.ndarray, timestamp: float):
        """保存帧数据(根据配置选择同步或异步)"""
        if self.use_async_save:
            try:
                # 放入异步保存队列
                self.save_queue.put_nowait((frame_id, rgb_image.copy(), pose, timestamp))
            except:
                # 队列满了,丢弃帧
                self.dropped_frames += 1
                if self.dropped_frames % 10 == 0:
                    logger.warning(f"保存队列满,已丢弃 {self.dropped_frames} 帧")
        else:
            # 同步保存
            self._save_frame_sync(frame_id, rgb_image, pose, timestamp)
    
    def _save_metadata_json(self):
        """保存元数据为JSON文件"""
        # 计算总路径长度（使用线程锁保护）
        with self.trajectory_lock:
            total_distance = 0.0
            for i in range(1, len(self.trajectory_data)):
                p1 = self.trajectory_data[i-1]
                p2 = self.trajectory_data[i]
                dist = np.sqrt((p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
                total_distance += dist
        
        duration = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / duration if duration > 0 else 0
        
        # 元数据
        metadata = {
            "session_info": {
                "session_timestamp": self.session_timestamp,
                "session_directory": str(self.session_dir),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat()
            },
            "recording_config": {
                "target_fps": self.recording_fps,
                "actual_fps": round(actual_fps, 2),
                "camera_resolution": self.camera_resolution,
                "async_save_enabled": self.use_async_save
            },
            "statistics": {
                "total_frames": self.frame_count,
                "dropped_frames": self.dropped_frames,
                "duration_seconds": round(duration, 2),
                "total_distance_meters": round(total_distance, 2)
            },
            "data_files": {
                "images_folder": "images",
                "trajectory_file": f"trajectory_{self.session_timestamp}.txt",
                "metadata_file": f"metadata_{self.session_timestamp}.json"
            }
        }
        
        # metadata JSON文件路径
        metadata_json_path = self.session_dir / f"metadata_{self.session_timestamp}.json"
        with open(metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 元数据已保存: {metadata_json_path}")
        
        # 打印统计信息
        logger.info(f"📊 记录统计:")
        logger.info(f"  - 会话时间戳: {self.session_timestamp}")
        logger.info(f"  - 总帧数: {self.frame_count}")
        logger.info(f"  - 丢帧数: {self.dropped_frames}")
        logger.info(f"  - 记录时长: {duration:.1f}秒")
        logger.info(f"  - 目标帧率: {self.recording_fps} Hz")
        logger.info(f"  - 实际帧率: {actual_fps:.1f} Hz")
        logger.info(f"  - 总路径长度: {total_distance:.2f}m")
        
        return metadata_json_path
    
    def record(self, duration: Optional[float] = None, show_preview: bool = True):
        """
        开始记录数据
        
        Args:
            duration: 记录时长(秒),None表示无限制
            show_preview: 是否显示预览窗口
        """
        logger.info("="*60)
        logger.info("开始高频数据记录")
        logger.info(f"记录频率: {self.recording_fps} Hz")
        logger.info(f"会话时间戳: {self.session_timestamp}")
        logger.info(f"保存目录: {self.session_dir}")
        logger.info(f"异步保存: {'开启' if self.use_async_save else '关闭'}")
        logger.info("按 'q' 键或 Ctrl+C 停止记录")
        logger.info("="*60)
        
        # 打开txt文件用于实时写入
        self.trajectory_file = open(self.trajectory_txt_path, 'w')
        self.trajectory_file.write("# Frame_ID\tX\tY\tYaw\n")  # 写入表头
        logger.info(f"✅ 轨迹文件已打开: {self.trajectory_txt_path}")
        
        # 启动异步保存线程
        if self.use_async_save:
            self.save_thread_running = True
            self.save_thread.start()
            logger.info("✅ 异步保存线程已启动")
        
        self.is_recording = True
        self.start_time = time.time()
        
        # 计算目标循环时间
        target_loop_time = 1.0 / self.recording_fps
        
        try:
            while self.is_recording:
                loop_start = time.time()
                
                # 检查时长限制
                if duration is not None and (loop_start - self.start_time) > duration:
                    logger.info(f"达到记录时长限制: {duration}秒")
                    break
                
                # 获取机器人位姿
                current_pose = self.robot.get_base_pose()
                if current_pose is None:
                    logger.warning("无法获取机器人位姿,跳过此帧")
                    time.sleep(target_loop_time)
                    continue
                
                current_pose = np.array(current_pose)
                
                # 获取头部相机图像
                frame = self.robot.camera.get_head_camera_frame()
                if frame is None or frame.get("rgb") is None:
                    logger.warning("无法获取相机图像,跳过此帧")
                    time.sleep(target_loop_time)
                    continue
                
                rgb_image = frame["rgb"]
                current_time = time.time()
                
                # 保存数据
                self._save_frame(self.frame_count, rgb_image, current_pose, current_time)
                self.frame_count += 1
                
                # 显示预览(可选)
                if show_preview:
                    display_image = rgb_image.copy()
                    
                    # 计算实际FPS
                    elapsed = current_time - self.start_time
                    actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    # 绘制信息
                    info_text = f"Session: {self.session_timestamp} | Frames: {self.frame_count} | FPS: {actual_fps:.1f}/{self.recording_fps}"
                    cv2.putText(display_image, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    pose_text = f"Pose: ({current_pose[0]:.2f}, {current_pose[1]:.2f}, {current_pose[2]:.2f})"
                    cv2.putText(display_image, pose_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if self.use_async_save:
                        queue_text = f"Save Queue: {self.save_queue.qsize()}/{self.save_queue.maxsize}"
                        color = (0, 255, 0) if self.save_queue.qsize() < 100 else (0, 165, 255)
                        cv2.putText(display_image, queue_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if self.dropped_frames > 0:
                        drop_text = f"Dropped: {self.dropped_frames}"
                        cv2.putText(display_image, drop_text, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imshow("Navigation Data Recording", display_image)
                    
                    # 按'q'退出
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("用户按下'q',停止记录")
                        break
                
                # 控制循环频率
                loop_time = time.time() - loop_start
                sleep_time = target_loop_time - loop_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 循环耗时超过目标时间
                    if self.frame_count % 100 == 0:
                        logger.warning(f"循环耗时 {loop_time*1000:.1f}ms > 目标 {target_loop_time*1000:.1f}ms")
                
        except KeyboardInterrupt:
            logger.info("\n收到Ctrl+C,停止记录")
        finally:
            self.is_recording = False
            
            # 关闭预览窗口
            if show_preview:
                cv2.destroyAllWindows()
            
            # 等待异步保存完成
            if self.use_async_save:
                logger.info("等待异步保存完成...")
                
                # 等待队列中的所有任务完成
                self.save_queue.join()
                logger.info("✅ 队列已清空")
                
                # 发送结束信号并等待线程退出
                self.save_thread_running = False
                self.save_thread.join(timeout=5)
                logger.info("✅ 异步保存线程已退出")
            
            # 关闭txt文件
            if self.trajectory_file and not self.trajectory_file.closed:
                self.trajectory_file.close()
                logger.info(f"✅ 轨迹文件已关闭: {self.trajectory_txt_path}")
            
            # 保存元数据JSON文件
            self._save_metadata_json()
            
            logger.info("="*60)
            logger.info("✅ 数据记录完成")
            logger.info(f"数据保存在: {self.session_dir}")
            logger.info(f"  - 图片: {self.image_dir}")
            logger.info(f"  - 轨迹: trajectory_{self.session_timestamp}.txt")
            logger.info(f"  - 元数据: metadata_{self.session_timestamp}.json")
            logger.info("="*60)
    
    def cleanup(self):
        """清理资源"""
        logger.info("停止相机流...")
        try:
            self.robot.camera.stop_stream()
        except:
            pass
        logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🤖 机器人导航数据记录工具 (Step 1: 高频记录)")
    print("="*60)
    
    # 使用默认机器人IP和相机SN
    robot_ip = "192.168.11.200"
    camera_sn = "233522073186"
    
    print(f"\n📡 机器人IP: {robot_ip}")
    print(f"📷 相机SN: {camera_sn}")
    
    # 设置参数
    print("\n📝 记录参数配置:")
    
    save_dir = input("数据保存根目录 [默认: ./navigation_raw_data]: ").strip()
    if not save_dir:
        save_dir = "./navigation_raw_data"
    
    recording_fps = input("记录频率(Hz) [默认: 20]: ").strip()
    recording_fps = int(recording_fps) if recording_fps else 20
    
    duration = input("记录时长(秒,留空表示无限制): ").strip()
    duration = float(duration) if duration else None
    
    use_async = input("使用异步保存? (y/n) [默认: y]: ").strip().lower()
    use_async = use_async != 'n'
    
    # 创建记录器
    try:
        recorder = NavigationDataRecorder(
            robot_ip=robot_ip,
            save_dir=save_dir,
            recording_fps=recording_fps,
            use_async_save=use_async,
            camera_sn=camera_sn
        )
        
        # 初始化相机
        recorder.initialize_camera()
        
        print("\n💡 提示: 现在可以开始遥控机器人移动")
        print(f"    脚本会以 {recording_fps} Hz 的频率记录所有数据")
        print(f"    数据将保存到: {recorder.session_dir}")
        input("按回车键开始记录...")
        
        # 开始记录
        recorder.record(duration=duration, show_preview=True)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
    finally:
        try:
            recorder.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()