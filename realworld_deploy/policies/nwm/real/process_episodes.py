"""
根据 record_data.py 采集的数据:
1) 修复轨迹和图片帧号缺失(丢帧补全: 复制上一帧 pose 和图片)
2) 从修复后的数据中按照行进距离下采样(每次采样距离在范围内随机),
   输出目录 epiosides_时间戳, 其中:
      - 图片命名为 0.jpg, 1.jpg, 2.jpg, ... (分辨率 320x240)
      - 轨迹保存为 traj_data.pkl, 为 dict:
          { "position": (N,2) np.ndarray, "yaw": (N,) np.ndarray }

使用示例:
    python process_episodes.py \
        --distance_step 0.25
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pickle


# ---------- 工具函数: 读取轨迹和图片帧号 ----------

def read_trajectory_txt(traj_path: Path) -> Dict[int, Tuple[float, float, float]]:
    """
    从 trajectory_*.txt 读取轨迹:
        每行: frame_id  x  y  yaw   (以空格或制表分隔)
    返回:
        {frame_id: (x, y, yaw)}
    """
    traj_dict: Dict[int, Tuple[float, float, float]] = {}

    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支持 tab 或 空格
            parts = re.split(r"[,\t ]+", line)
            if len(parts) < 4:
                continue
            try:
                frame_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                yaw = float(parts[3])
                traj_dict[frame_id] = (x, y, yaw)
            except ValueError:
                # 表头或格式异常行跳过
                continue

    if not traj_dict:
        raise RuntimeError(f"轨迹文件为空或无法解析: {traj_path}")
    return traj_dict


def read_image_frames(images_dir: Path) -> List[int]:
    """
    读取 images 目录下的 jpg 文件, 解析帧号(文件名为 000000.jpg 形式),
    返回排序后的帧号列表.
    """
    frame_ids: List[int] = []
    for img_path in images_dir.glob("*.jpg"):
        try:
            frame_ids.append(int(img_path.stem))
        except ValueError:
            # 忽略不符合帧号格式的文件
            continue
    frame_ids.sort()
    if not frame_ids:
        raise RuntimeError(f"目录中未找到 jpg 图片: {images_dir}")
    return frame_ids


# ---------- 功能1: 修复轨迹和图片丢帧 ----------

def fix_missing_frames(
    images_dir: Path,
    traj_dict: Dict[int, Tuple[float, float, float]],
) -> List[Tuple[int, float, float, float]]:
    """
    修复轨迹 + 图片:
      - 轨迹: 在 [min_frame, max_frame] 范围内补齐缺失帧, 用上一帧 pose 复制
      - 图片: 若缺失某帧 jpg, 用上一帧图片复制成该帧

    注意: 直接在原 images 目录中补齐缺失图片.
    返回:
      fixed_traj: [(frame_id, x, y, yaw), ...] (按帧号升序)
    """
    img_frame_ids = read_image_frames(images_dir)
    existing_img_set = set(img_frame_ids)

    # 轨迹范围
    sorted_traj_frames = sorted(traj_dict.keys())
    min_frame = sorted_traj_frames[0]
    max_frame = sorted_traj_frames[-1]

    # 准备首帧图片和 pose
    first_frame_for_img = min(img_frame_ids)
    first_img_path = images_dir / f"{first_frame_for_img:06d}.jpg"
    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        raise RuntimeError(f"无法读取首帧图片: {first_img_path}")

    # 若最小轨迹帧在图片之前/之后, 统一从 min_frame 开始处理
    fixed_traj: List[Tuple[int, float, float, float]] = []

    # 初始 pose: 轨迹中最小帧的 pose
    last_pose = traj_dict[min_frame]

    # last_img 默认使用可读到的第一张图片
    last_img = first_img.copy()

    for fid in range(min_frame, max_frame + 1):
        # -------- 轨迹修复 --------
        if fid in traj_dict:
            last_pose = traj_dict[fid]
        # 否则沿用 last_pose

        x, y, yaw = last_pose
        fixed_traj.append((fid, x, y, yaw))

        # -------- 图片修复 --------
        img_path = images_dir / f"{fid:06d}.jpg"
        if fid in existing_img_set:
            # 该帧名存在, 重新读取并更新 last_img (如果读取失败则用旧图覆盖)
            img = cv2.imread(str(img_path))
            if img is None:
                cv2.imwrite(str(img_path), last_img)
            else:
                last_img = img
        else:
            # 该帧图片缺失, 直接写入上一帧图片
            cv2.imwrite(str(img_path), last_img)

    return fixed_traj


# ---------- 功能2: 按行进距离下采样 ----------

def downsample_by_distance(
    fixed_traj: List[Tuple[int, float, float, float]],
    min_distance_step: float,
    max_distance_step: float
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    按行进距离每 distance_step 米选一帧:
      - 从起点开始累计欧氏距离, 每 >= step 记录一帧
      - 起点也会被记录

    输入:
      fixed_traj: [(frame_id, x, y, yaw), ...] 按 frame_id 升序

    返回:
      selected_frame_ids: List[int]
      positions: (N,2) np.ndarray
      yaws: (N,) np.ndarray
    """
    if not fixed_traj:
        raise RuntimeError("fixed_traj 为空")
    
    rng = np.random.default_rng()
    
    def sample_step() -> float:
        if max_distance_step == min_distance_step:
            return float(min_distance_step)
        return float(rng.uniform(min_distance_step, max_distance_step))

    selected_frames: List[int] = []
    pos_list: List[Tuple[float, float]] = []
    yaw_list: List[float] = []
    distance_list: List[float] = []

    # 起点
    fid0, x0, y0, yaw0 = fixed_traj[0]
    last_x, last_y = x0, y0
    accumulated_dist = 0.0
    distance_step = sample_step()

    selected_frames.append(fid0)
    pos_list.append((x0, y0))
    yaw_list.append(yaw0)

    for fid, x, y, yaw in fixed_traj[1:]:
        dx = x - last_x
        dy = y - last_y
        step_dist = float(np.sqrt(dx * dx + dy * dy))

        accumulated_dist += step_dist
        last_x, last_y = x, y

        if accumulated_dist >= distance_step:
            selected_frames.append(fid)
            pos_list.append((x, y))
            yaw_list.append(yaw)
            distance_list.append(accumulated_dist)
            accumulated_dist = 0.0
            distance_step = sample_step()
            

    positions = np.asarray(pos_list, dtype=np.float32)
    yaws = np.asarray(yaw_list, dtype=np.float32)
    return selected_frames, positions, yaws, np.mean(distance_list)


# ---------- 输出: 复制图片 + 保存 pkl ----------

def copy_and_resize_selected_images(
    images_dir: Path,
    out_dir: Path,
    selected_frames: List[int],
    out_size: Tuple[int, int] = (320, 240),
):
    """
    将选中的帧图片:
      - 从 images_dir/xxxxx.jpg 读取
      - 缩放到 out_size (w,h) = (320,240)
      - 保存到 out_dir/0.jpg, 1.jpg, 2.jpg, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = out_size

    for idx, fid in enumerate(selected_frames):
        src = images_dir / f"{fid:06d}.jpg"
        if not src.exists():
            raise RuntimeError(f"下采样所需图片缺失: {src}")
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError(f"无法读取图片: {src}")

        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        dst = out_dir / f"{idx}.jpg"
        cv2.imwrite(str(dst), resized)


def save_episodes_pkl(
    out_dir: Path,
    positions: np.ndarray,
    yaws: np.ndarray,
    pkl_name: str = "traj_data.pkl",
):
    """
    保存下采样轨迹为 dict 的 pkl 文件, 无序号, 固定文件名.
    结构为:
        { "position": (N,2) array, "yaw": (N,) array }
    """
    data = {
        "position": positions,
        "yaw": yaws,
    }
    out_path = out_dir / pkl_name
    with out_path.open("wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] 轨迹 pkl 已保存: {out_path}")
    print(f"       数据类型: {type(data)}")
    print(f"       键: {list(data.keys())}")
    print(f"       position shape: {positions.shape}, yaw shape: {yaws.shape}")


# ---------- 主流程 ----------

def process_session(
    session_dir: Path,
    out_root_dir: Path,
    min_distance_step: float = 0.25,
    max_distance_step: float = 0.25
):
    """
    对单个会话目录执行:
      1) 修复轨迹 & 图片帧号 (功能1)
      2) 按距离下采样并导出到指定输出根目录下对应子目录

    参数:
      session_dir: 单次会话原始数据目录, 如 /.../navigation_raw_data/20250203_123456
      out_root_dir: 处理后数据的根目录, 如 /.../navigation_process_data
      min_distance_step: 最小下采样步长(米)
    """
    if not session_dir.exists():
        raise RuntimeError(f"session_dir 不存在: {session_dir}")

    # 1. 轨迹文件 trajectory_*.txt
    traj_files = sorted(session_dir.glob("trajectory_*.txt"))
    if not traj_files:
        raise RuntimeError(f"未在 {session_dir} 找到 trajectory_*.txt")
    traj_path = traj_files[0]

    # 2. 图片目录 images
    images_dir = session_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"未在 {session_dir} 找到 images 目录")

    print(f"[INFO] 使用轨迹文件: {traj_path}")
    print(f"[INFO] 图片目录: {images_dir}")

    # --- 功能1: 修复丢帧 ---
    traj_dict = read_trajectory_txt(traj_path)
    fixed_traj = fix_missing_frames(images_dir, traj_dict)
    print(f"[INFO] 修复后总帧数: {len(fixed_traj)} "
          f"(frame range: {fixed_traj[0][0]} ~ {fixed_traj[-1][0]})")

    # --- 功能2: 按距离下采样 ---
    selected_frames, positions, yaws, mean_dist = downsample_by_distance(
        fixed_traj, min_distance_step=min_distance_step, max_distance_step=max_distance_step
    )
    print(f"[INFO] 按 最小: {min_distance_step}m，最大：{max_distance_step}m 下采样后关键帧数: {len(selected_frames)}")
    print(f"[INFO] 平均距离: {mean_dist}m")

    # 输出目录: 在 out_root_dir 下创建与 session_dir 同名的子目录
    out_root_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_root_dir / session_dir.name
    print(f"[INFO] 输出目录: {out_dir}")

    # 复制 + 缩放图片
    copy_and_resize_selected_images(
        images_dir=images_dir,
        out_dir=out_dir,
        selected_frames=selected_frames,
        out_size=(320, 240),
    )

    # 保存 pkl
    save_episodes_pkl(out_dir, positions, yaws, pkl_name="traj_data.pkl")

    print("[INFO] 处理完成.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 record_data.py 采集的数据进行: "
                    "1) 丢帧修复 2) 按行进距离下采样, "
                    "批量处理 navigation_raw_data 下的所有会话目录"
    )
    parser.add_argument(
        "--min_distance_step",
        type=float,
        default=0.2,
        help="按行进距离下采样的步长(米), 默认 0.2",
    )
    parser.add_argument(
        "--max_distance_step",
        type=float,
        default=0.3,
        help="按行进距离下采样的步长(米), 默认 0.3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    min_distance_step = float(args.min_distance_step)
    max_distance_step = float(args.max_distance_step)

    # 固定原始数据根目录与输出根目录
    raw_root = Path("D:\\embodiedAI\\DISCOVERSE\\policies\\nwm\\real\\navigation_raw_data").expanduser().resolve()
    out_root = Path("D:\\embodiedAI\\DISCOVERSE\\policies\\nwm\\real\\navigation_process_data").expanduser().resolve()

    if not raw_root.exists():
        raise RuntimeError(f"原始数据根目录不存在: {raw_root}")

    print(f"[INFO] 原始数据根目录: {raw_root}")
    print(f"[INFO] 处理后数据根目录: {out_root}")
    print(f"[INFO] 下采样步长: {min_distance_step} m - {max_distance_step} m")

    # 遍历所有子目录(假定为时间戳目录)
    for session_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        print(f"\n[INFO] 开始处理会话目录: {session_dir}")
        try:
            process_session(
                session_dir=session_dir,
                out_root_dir=out_root,
                min_distance_step=min_distance_step,
                max_distance_step=max_distance_step
            )
        except Exception as e:
            # 打印错误但不中断其它目录处理
            print(f"[ERROR] 处理目录 {session_dir} 失败: {e}")

    print("\n[INFO] 全部会话目录处理完成.")


if __name__ == "__main__":
    main()
