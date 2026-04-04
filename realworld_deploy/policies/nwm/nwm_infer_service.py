"""
服务器端推理服务 - WebSocket服务器
接收机器人端的4帧观测+1帧目标图像，使用用户模型进行推理
保存预测图片到服务器端，返回轨迹给机器人端
"""

import asyncio
import websockets
import json
import numpy as np
import torch
from PIL import Image
import io
import base64
import os
from datetime import datetime
from torchvision import transforms
import torchvision.transforms.functional as TF
import logging
import yaml
import argparse
from pathlib import Path

# 导入用户的模型模块
from planning import WM_Planning_Policy
from misc import unnormalize_data, calculate_delta_yaw

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_ASPECT_RATIO = (4 / 3)

class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.ar)))
        else:
            img = TF.center_crop(img, (int(w / self.ar), w))
        return img

# 图像预处理
imgTransform = transforms.Compose([
    CenterCropAR(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

# 反归一化
unnormalize_fn = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)


class SimpleArgs:
    """从配置文件构建模型参数"""
    def __init__(self, config):
        self.datasets = config['model']['dataset_name']
        self.num_samples = config['planning']['num_samples']
        self.rollout_stride = config['planning']['rollout_stride']
        self.topk = config['planning']['topk']
        self.opt_steps = config['planning']['opt_steps']
        self.num_repeat_eval = config['planning']['num_repeat_eval']
        self.obs_steps = config['observation']['context_size']
        self.lora_dir = config['model']['lora_dir']
        self.diffusion_infer_mode = config['model']['diffusion_infer_mode']
        
        class WMEval:
            pass
        
        self.wm_eval = WMEval()
        self.wm_eval.exp = config['model']['config_path']
        self.wm_eval.ckp = config['model']['checkpoint']
        
        # 加载metric_waypoint_spacing
        with open("config/data_config.yaml", "r") as f:
            data_config = yaml.safe_load(f)
        self.wm_eval.metric_waypoint_spacing = data_config[self.datasets]['metric_waypoint_spacing']
        
        self.work_dir = None


class InferenceServer:
    """推理服务器类"""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        初始化推理服务器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录（保存预测图片）
        """
        logger.info("初始化推理服务器...")
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录
        self.pred_images_dir = os.path.join(self.output_dir, 'predicted_images')
        self.obs_images_dir = os.path.join(self.output_dir, 'obs_images')
        self.goal_images_dir = os.path.join(self.output_dir, 'goal_images')
        self.trajectories_dir = os.path.join(self.output_dir, 'trajectories')
        os.makedirs(self.pred_images_dir, exist_ok=True)
        os.makedirs(self.obs_images_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)
        os.makedirs(self.goal_images_dir, exist_ok=True)
        
        # 加载模型
        logger.info("加载推理模型...")
        model_args = SimpleArgs(self.config)
        model_args.work_dir = self.output_dir
        self.agent = WM_Planning_Policy(model_args)
        logger.info("模型加载完成")
        
        # 推理计数器
        self.inference_count = 0
        
        # 加载action stats
        with open("config/data_config.yaml", "r") as f:
            data_config = yaml.safe_load(f)
        self.ACTION_STATS_TORCH = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])
    
    def decode_image(self, base64_str: str) -> Image.Image:
        """解码base64图像"""
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        return img
    
    def run_inference(self, obs_seq: torch.Tensor, goal_tensor: torch.Tensor):
        """执行模型推理，逻辑与 real_infer 中保持一致"""
        # 先用策略生成动作与整体偏航角
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred_actions, _, pred_images = self.agent.generate_actions(obs_seq, goal_tensor)

        return pred_images, pred_actions
        

    def save_images(self, images: torch.Tensor, iteration: int, iter_dir: str):
        """保存图像序列"""
        os.makedirs(iter_dir, exist_ok=True)
        
        images = images.squeeze(0)
        for i in range(images.shape[0]):
            # images 可能为 bfloat16，NumPy 不支持；先转换为 float32 再反归一化
            img_tensor = images[i].to(torch.float32)
            # 反归一化
            img_tensor = unnormalize_fn(img_tensor)
            img_tensor = torch.clamp(img_tensor, 0, 1)
            # 转换为numpy
            img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # 保存
            Image.fromarray(img_array).save(os.path.join(iter_dir, f'{i}.png'))
            
    def save_trajectory(self, pred_actions: torch.Tensor, iteration: int):
        """保存轨迹数据"""
        traj_file = os.path.join(self.trajectories_dir, f'trajectory_{iteration:04d}.txt')
        
        with open(traj_file, 'w') as f:
            f.write(f"# Inference {iteration}\n")
            f.write("# Format: x(m) y(m) yaw(rad)\n")
            actions = pred_actions.squeeze(0).cpu().numpy()  # [pred_horizon, 3]
            cumulative_yaw = 0.0
            for i in range(len(actions)):
                delta_x, delta_y, delta_yaw = actions[i]
                cumulative_yaw += delta_yaw
                f.write(f"{delta_x:.6f} {delta_y:.6f} {delta_yaw:6f} {cumulative_yaw:.6f}\n")
        
        logger.info(f"已保存轨迹到: {traj_file}")
    
    async def handle_client(self, websocket):
        """处理客户端连接（兼容当前 websockets 版本，仅接收 websocket 对象）"""
        client_addr = websocket.remote_address
        logger.info(f"客户端连接: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    # 解析接收的数据
                    data = json.loads(message)
                    
                    if data['type'] == 'inference_request':
                        logger.info(f"收到推理请求 (推理计数: {self.inference_count})")
                        
                        # 解码图像
                        obs_images = []
                        for i in range(4):
                            img = self.decode_image(data[f'obs_{i}'])
                            obs_tensor = imgTransform(img)
                            obs_images.append(obs_tensor)
                        
                        goal_img = self.decode_image(data['goal'])
                        goal_tensor = imgTransform(goal_img).unsqueeze(0).unsqueeze(0)  # [1,1,3,224,224]
                        
                        # 构建观测序列
                        obs_seq = torch.stack(obs_images).unsqueeze(0)  # [1,4,3,224,224]
                        
                        # 执行推理
                        logger.info("执行模型推理...")
                        pred_images, pred_actions = self.run_inference(obs_seq, goal_tensor)

                        # 保存目标图像
                        iter_dir = os.path.join(self.goal_images_dir, f'inference_{self.inference_count:04d}')
                        self.save_images(goal_tensor, self.inference_count, iter_dir) 

                        # 保存观察图像
                        iter_dir = os.path.join(self.obs_images_dir, f'inference_{self.inference_count:04d}')
                        self.save_images(obs_seq, self.inference_count, iter_dir)
                        
                        # 保存预测图像
                        iter_dir = os.path.join(self.pred_images_dir, f'inference_{self.inference_count:04d}')
                        self.save_images(pred_images, self.inference_count, iter_dir)
                        
                        # 保存轨迹
                        self.save_trajectory(pred_actions, self.inference_count)
                        
                        # 准备响应数据
                        actions = pred_actions.squeeze(0).cpu().numpy()  # [pred_horizon, 2]
                        
                        # 构建轨迹列表
                        delta_actions = []
                        for i in range(len(actions)):
                            dx, dy, dtheta = actions[i]
                            delta_actions.append({
                                'dx': float(dx),
                                'dy': float(dy),
                                'dtheta': float(dtheta),
                            })
                        
                        # 发送响应
                        response = {
                            'type': 'inference_response',
                            'status': 'success',
                            'inference_id': self.inference_count,
                            'delta_actions': delta_actions,
                            'num_waypoints': len(delta_actions)
                        }
                        
                        await websocket.send(json.dumps(response))
                        logger.info(f"已返回轨迹 (共{len(delta_actions)}个航点)")
                        
                        self.inference_count += 1
                    
                    elif data['type'] == 'ping':
                        # 心跳响应
                        await websocket.send(json.dumps({'type': 'pong'}))
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"处理请求时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接: {client_addr}")
        except Exception as e:
            logger.error(f"连接错误: {e}")
            import traceback
            traceback.print_exc()
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """启动WebSocket服务器"""
        logger.info("="*60)
        logger.info("启动推理服务器")
        logger.info("="*60)
        logger.info(f"监听地址: {host}:{port}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"预测图像: {self.pred_images_dir}")
        logger.info(f"轨迹数据: {self.trajectories_dir}")
        logger.info("="*60)
        
        # max_size=None 允许接收大于默认1MB的消息帧（我们要传5张base64图像）
        async with websockets.serve(self.handle_client, host, port, max_size=None):
            logger.info("服务器已启动，等待连接...")
            await asyncio.Future()  # 永久运行


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='推理服务器 - 接收观测并返回轨迹')
    parser.add_argument('--config', type=str, default='real_infer.yaml',
                       help='配置文件路径')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务主机地址')
    parser.add_argument('--output-dir', type=str, default='server_outputs',
                       help='输出目录')
    args = parser.parse_args()
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # 创建并启动服务器
    server = InferenceServer(args.config, output_dir)
    
    try:
        asyncio.run(server.start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("\n服务器被用户中断")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ============================================================
# 使用说明
# ============================================================
#
# 启动服务器:
#   python nwm_infer_service.py \
#     --config /root/autodl-tmp/jason/DISCOVERSE/policies/nwm/config/real_infer.yaml \
#     --port 8000 \
#     --output-dir server_outputs
#   python nwm_infer_service.py --config /root/autodl-tmp/jason/DISCOVERSE/policies/nwm/config/real_infer.yaml --port 8000 --output-dir server_outputs
# 说明:
# 1. 服务器接收包含4帧观测+1帧目标图像的JSON数据
# 2. 执行模型推理生成轨迹
# 3. 保存预测图像到 server_outputs_<timestamp>/predicted_images/
# 4. 保存轨迹数据到 server_outputs_<timestamp>/trajectories/
# 5. 返回轨迹给客户端
# ============================================================