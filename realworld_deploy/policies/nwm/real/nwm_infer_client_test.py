"""
NWM推理服务测试客户端
用于测试nwm_infer_service.py服务器的连接和推理功能
"""

import asyncio
import websockets
import json
import numpy as np
from PIL import Image
import io
import base64
import argparse
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NWMTestClient:
    """NWM推理服务测试客户端"""
    
    def __init__(self, server_url: str, output_dir: str = "client_test_outputs", recv_timeout: float = 600.0):
        """
        初始化测试客户端
        
        Args:
            server_url: 服务器WebSocket地址 (例如: ws://服务器IP:8000)
            output_dir: 本地输出目录
        """
        self.server_url = server_url
        self.output_dir = output_dir
        self.recv_timeout = recv_timeout
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录
        self.sent_images_dir = os.path.join(self.output_dir, 'sent_images')
        self.trajectories_dir = os.path.join(self.output_dir, 'received_trajectories')
        os.makedirs(self.sent_images_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)
        
        logger.info(f"测试客户端初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
    
    def generate_random_image(self, width: int = 640, height: int = 480) -> Image.Image:
        """
        生成随机RGB图像
        
        Args:
            width: 图像宽度
            height: 图像高度
        
        Returns:
            PIL Image对象
        """
        # 生成随机RGB数据
        random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(random_array, 'RGB')
        return img
    
    def generate_pattern_image(self, pattern_type: str = "gradient", width: int = 640, height: int = 480) -> Image.Image:
        """
        生成带图案的测试图像（更易于区分不同帧）
        
        Args:
            pattern_type: 图案类型 ("gradient", "checkerboard", "random")
            width: 图像宽度
            height: 图像高度
        
        Returns:
            PIL Image对象
        """
        if pattern_type == "gradient":
            # 渐变图案
            x = np.linspace(0, 255, width, dtype=np.uint8)
            y = np.linspace(0, 255, height, dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            img_array = np.stack([xx, yy, 128 * np.ones_like(xx)], axis=-1)
        
        elif pattern_type == "checkerboard":
            # 棋盘图案
            block_size = 50
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if ((i // block_size) + (j // block_size)) % 2 == 0:
                        img_array[i:i+block_size, j:j+block_size] = [255, 255, 255]
                    else:
                        img_array[i:i+block_size, j:j+block_size] = [100, 150, 200]
        
        else:  # random
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        return Image.fromarray(img_array, 'RGB')
    
    def encode_image_to_base64(self, img: Image.Image) -> str:
        """
        将PIL Image编码为base64字符串
        
        Args:
            img: PIL Image对象
        
        Returns:
            base64编码的字符串
        """
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    
    def save_test_images(self, obs_images: list, goal_image: Image.Image, test_id: int):
        """保存发送的测试图像"""
        test_dir = os.path.join(self.sent_images_dir, f'test_{test_id:04d}')
        os.makedirs(test_dir, exist_ok=True)
        
        # 保存观测图像
        for i, img in enumerate(obs_images):
            img.save(os.path.join(test_dir, f'obs_{i}.png'))
        
        # 保存目标图像
        goal_image.save(os.path.join(test_dir, f'goal.png'))
        
        logger.info(f"已保存测试图像到: {test_dir}")
    
    def save_trajectory(self, trajectory: list, metadata: dict, test_id: int):
        """保存接收到的轨迹数据"""
        traj_file = os.path.join(self.trajectories_dir, f'trajectory_{test_id:04d}.txt')
        
        with open(traj_file, 'w') as f:
            f.write(f"# Test {test_id}\n")
            f.write(f"# Inference ID: {metadata.get('inference_id', 'N/A')}\n")
            f.write(f"# Total Yaw: {metadata.get('total_yaw', 'N/A')} rad\n")
            f.write(f"# Num Waypoints: {metadata.get('num_waypoints', len(trajectory))}\n")
            f.write("# Format: x(m) y(m) yaw(rad)\n")
            f.write("\n")
            
            for waypoint in trajectory:
                x = waypoint['x']
                y = waypoint['y']
                yaw = waypoint['yaw']
                f.write(f"{x:.6f} {y:.6f} {yaw:.6f}\n")
        
        logger.info(f"已保存轨迹到: {traj_file}")
    
    async def send_inference_request(self, websocket, test_id: int = 0, 
                                     use_pattern: bool = True):
        """
        发送推理请求
        
        Args:
            websocket: WebSocket连接
            test_id: 测试ID
            use_pattern: 是否使用图案图像（更易区分）
        """
        logger.info(f"准备测试 #{test_id}...")
        
        # 生成测试图像
        obs_images = []
        if use_pattern:
            patterns = ["gradient", "checkerboard", "random", "gradient"]
            for i, pattern in enumerate(patterns):
                img = self.generate_pattern_image(pattern)
                obs_images.append(img)
                logger.info(f"  生成观测图像 {i}: {pattern}图案")
            goal_image = self.generate_pattern_image("random")
            logger.info(f"  生成目标图像: random图案")
        else:
            for i in range(4):
                img = self.generate_random_image()
                obs_images.append(img)
                logger.info(f"  生成观测图像 {i}: 随机图像")
            goal_image = self.generate_random_image()
            logger.info(f"  生成目标图像: 随机图像")
        
        # 保存测试图像
        self.save_test_images(obs_images, goal_image, test_id)
        
        # 编码图像
        logger.info("编码图像为base64...")
        request_data = {
            'type': 'inference_request',
            'test_id': test_id,
            'timestamp': datetime.now().isoformat()
        }
        
        for i, img in enumerate(obs_images):
            request_data[f'obs_{i}'] = self.encode_image_to_base64(img)
        
        request_data['goal'] = self.encode_image_to_base64(goal_image)
        
        # 发送请求
        logger.info("发送推理请求到服务器...")
        await websocket.send(json.dumps(request_data))
        logger.info("✓ 请求已发送")
        
        # 接收响应
        logger.info(f"等待服务器响应... (超时时间 {self.recv_timeout}s)")
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=self.recv_timeout)
        except asyncio.TimeoutError:
            logger.error(f"✗ 等待服务器响应超时 ({self.recv_timeout}s)")
            return False

        response_data = json.loads(response)
        
        if response_data['type'] == 'inference_response':
            if response_data['status'] == 'success':
                logger.info("✓ 收到成功响应")
                logger.info(f"  推理ID: {response_data['inference_id']}")
                logger.info(f"  航点数量: {response_data['num_waypoints']}")
                logger.info(f"  总偏航角: {response_data['total_yaw']:.4f} rad")
                
                # 显示轨迹预览
                trajectory = response_data['trajectory']
                logger.info("  轨迹预览 (前3个航点):")
                for i, wp in enumerate(trajectory[:3]):
                    logger.info(f"    [{i}] x={wp['x']:.4f}, y={wp['y']:.4f}, yaw={wp['yaw']:.4f}")
                if len(trajectory) > 3:
                    logger.info(f"    ... (共{len(trajectory)}个航点)")
                
                # 保存轨迹
                metadata = {
                    'inference_id': response_data['inference_id'],
                    'total_yaw': response_data['total_yaw'],
                    'num_waypoints': response_data['num_waypoints']
                }
                self.save_trajectory(trajectory, metadata, test_id)
                
                return True
            else:
                logger.error(f"✗ 服务器返回错误: {response_data.get('message', 'Unknown error')}")
                return False
        
        elif response_data['type'] == 'error':
            logger.error(f"✗ 服务器错误: {response_data['message']}")
            return False
        
        else:
            logger.error(f"✗ 未知响应类型: {response_data['type']}")
            return False
    
    async def test_connection(self):
        """测试与服务器的连接"""
        logger.info("="*60)
        logger.info("开始连接测试")
        logger.info("="*60)
        logger.info(f"目标服务器: {self.server_url}")
        
        try:
            async with websockets.connect(self.server_url, ping_interval=None) as websocket:
                logger.info("✓ WebSocket连接成功")
                
                # 发送ping测试
                logger.info("发送心跳测试...")
                await websocket.send(json.dumps({'type': 'ping'}))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if response_data['type'] == 'pong':
                    logger.info("✓ 心跳响应正常")
                    return True
                else:
                    logger.warning("⚠ 心跳响应异常")
                    return False
        
        except Exception as e:
            logger.error(f"✗ 连接失败: {e}")
            return False
    
    async def run_tests(self, num_tests: int = 3, use_pattern: bool = True):
        """
        运行多次推理测试
        
        Args:
            num_tests: 测试次数
            use_pattern: 是否使用图案图像
        """
        logger.info("="*60)
        logger.info("开始推理测试")
        logger.info("="*60)
        logger.info(f"目标服务器: {self.server_url}")
        logger.info(f"测试次数: {num_tests}")
        logger.info(f"图像类型: {'图案' if use_pattern else '随机'}")
        logger.info("="*60)
        
        success_count = 0
        
        try:
            async with websockets.connect(self.server_url, ping_interval=None) as websocket:
                logger.info("✓ WebSocket连接成功")
                logger.info("")
                
                for i in range(num_tests):
                    logger.info(f"{'='*60}")
                    logger.info(f"测试 {i+1}/{num_tests}")
                    logger.info(f"{'='*60}")
                    
                    success = await self.send_inference_request(websocket, i, use_pattern)
                    
                    if success:
                        success_count += 1
                        logger.info(f"✓ 测试 {i+1} 成功")
                    else:
                        logger.error(f"✗ 测试 {i+1} 失败")
                    
                    logger.info("")
                    
                    # 测试间隔
                    if i < num_tests - 1:
                        await asyncio.sleep(1)
                
                # 总结
                logger.info("="*60)
                logger.info("测试完成")
                logger.info("="*60)
                logger.info(f"成功: {success_count}/{num_tests}")
                logger.info(f"失败: {num_tests - success_count}/{num_tests}")
                logger.info(f"成功率: {success_count/num_tests*100:.1f}%")
                logger.info("="*60)
        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket错误: {e}")
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NWM推理服务测试客户端')
    parser.add_argument('--host', type=str, required=True,
                       help='服务器IP地址')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务器端口 (默认: 8000)')
    parser.add_argument('--num-tests', type=int, default=3,
                       help='测试次数 (默认: 3)')
    parser.add_argument('--output-dir', type=str, default='client_test_outputs',
                       help='输出目录 (默认: client_test_outputs)')
    parser.add_argument('--test-connection', action='store_true',
                       help='仅测试连接')
    parser.add_argument('--random-images', action='store_true',
                       help='使用完全随机图像（默认使用图案图像）')
    
    args = parser.parse_args()
    
    # 构建服务器URL
    server_url = f"ws://{args.host}:{args.port}"
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # 创建客户端
    client = NWMTestClient(server_url, output_dir)
    
    try:
        if args.test_connection:
            # 仅测试连接
            success = await client.test_connection()
            if success:
                logger.info("✓ 连接测试通过")
            else:
                logger.error("✗ 连接测试失败")
        else:
            # 运行完整测试
            await client.run_tests(
                num_tests=args.num_tests,
                use_pattern=not args.random_images
            )
    
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 使用说明
#  ssh -p 40034 -L 8000:127.0.0.1:8000 root@connect.westd.seetacloud.com
# ============================================================
#
# 【基础测试】连接到服务器并运行3次推理测试:
#
#   python nwm_test_client.py --host <服务器IP>
#
# 【完整示例】指定所有参数:
#
#   python nwm_test_client.py \
#     --host 192.168.1.100 \
#     --port 8000 \
#     --num-tests 5 \
#     --output-dir my_test_outputs
#
# 【仅测试连接】检查服务器是否可达:
#
#   python nwm_test_client.py \
#     --host 192.168.1.100 \
#     --test-connection
#
# 【使用随机图像】(默认使用图案图像更易区分):
#
#   python nwm_test_client.py \
#     --host 192.168.1.100 \
#     --random-images
#
# ============================================================
# 输出文件
# ============================================================
#
# 生成的输出文件结构:
#
# client_test_outputs_<timestamp>/
# ├── sent_images/              # 发送的测试图像
# │   ├── test_0000/
# │   │   ├── obs_0.png        # 观测图像0
# │   │   ├── obs_1.png        # 观测图像1
# │   │   ├── obs_2.png        # 观测图像2
# │   │   ├── obs_3.png        # 观测图像3
# │   │   └── goal.png         # 目标图像
# │   ├── test_0001/
# │   └── ...
# └── received_trajectories/    # 接收的轨迹数据
#     ├── trajectory_0000.txt
#     ├── trajectory_0001.txt
#     └── ...
#
# ============================================================
# 测试流程
# ============================================================
#
# 1. 建立WebSocket连接
# 2. 生成4帧观测图像 + 1帧目标图像
# 3. 将图像编码为base64
# 4. 发送推理请求到服务器
# 5. 接收服务器返回的轨迹数据
# 6. 保存发送的图像和接收的轨迹
# 7. 重复指定次数
#
# ============================================================
# 故障排查
# ============================================================
#
# 如果连接失败:
# 1. 检查服务器IP和端口是否正确
# 2. 检查服务器防火墙是否开放8000端口
# 3. 检查网络连接是否正常
# 4. 使用 --test-connection 先测试连接
#
# 如果推理失败:
# 1. 检查服务器日志输出
# 2. 检查服务器模型是否正确加载
# 3. 查看返回的错误信息
#
# ============================================================