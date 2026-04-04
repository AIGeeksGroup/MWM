
"""
NWM 实机推理客户端（MMK2 机器人）

功能：
1) 目标图像采集：
   - 提示用户推动机器人到目标位置
   - 终端输入 y 回车或在预览窗口按 y 确认
   - 保存目标图像到 goals_dir

2) 启动客户端后：
   - 让用户选择已采集的目标
   - 提示用户推动机器人到自定义起点
   - 实时显示头部相机画面
   - 确认后发送最近4帧观测 + 目标帧到服务端（WebSocket）

3) 接收服务端动作序列：
   - 打印推理耗时
   - 保存返回的轨迹/相对动作
   - 提示用户确认后执行导航
   - 使用 SDK 的 move_forward / turn_left / turn_right 执行相对动作
   - 每步完成后保存当前观察帧与位姿

协议（兼容 nwm_infer_client_test.py）：
- 发送 JSON：
    {
      "type": "inference_request",
      "timestamp": "...",
      "obs_0": "<base64 PNG>",
      "obs_1": ...,
      "obs_2": ...,
      "obs_3": ...,
      "goal": "<base64 PNG>",
      "meta": { ... optional ... }
    }
- 接收 JSON：
    { "type":"inference_response", "status":"success", ... }
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import websockets

# MMK2 SDK
from mmk2_sdk import MMK2Robot, RobotMode


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nwm_real_client")


# ---------------------------
# Helpers
# ---------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def b64_png_from_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_from_frame_rgb(frame_rgb: np.ndarray) -> Image.Image:
    # frame_rgb expected shape (H, W, 3) uint8
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_rgb, mode="RGB")


def normalize_angle(angle: float) -> float:
    """Normalize to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest angle difference a-b in [-pi, pi)."""
    return normalize_angle(a - b)


class InputWaiter:
    """Background stdin waiter: user types 'y' then Enter to confirm."""
    def __init__(self):
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self):
        self._event.clear()

    def _run(self):
        while not self._stop.is_set():
            try:
                s = input().strip().lower()
            except EOFError:
                return
            if s == "y":
                self._event.set()


@dataclass
class GoalItem:
    name: str
    path: Path


# ---------------------------
# Camera + robot wrapper
# ---------------------------

class RobotIO:
    def __init__(
        self,
        robot_ip: str,
        camera_sn: str,
        camera_resolution: str,
        assume_frame_is_rgb: bool = True,
    ):
        self.robot_ip = robot_ip
        self.camera_sn = camera_sn
        self.camera_resolution = camera_resolution
        self.assume_frame_is_rgb = assume_frame_is_rgb

        logger.info(f"正在连接机器人： IP={robot_ip}")
        self.robot = MMK2Robot(ip=robot_ip, mode=RobotMode.REAL)
        if not self.robot.is_connected():
            raise ConnectionError("机器人连接失败")
        logger.info("机器人已连接")

    def init_camera(self):
        logger.info(f"初始化头部相机： 序列号={self.camera_sn}, 分辨率配置={self.camera_resolution}")
        camera_config = {
            "head_camera": {
                "camera_type": "REALSENSE",
                "serial_no": f"'{self.camera_sn}'",
                "rgb_camera.color_profile": self.camera_resolution,
                "enable_depth": "false",
                "align_depth.enable": "false",
            }
        }
        self.robot.camera.set_camera_config(camera_config)
        self.robot.camera.start_stream(["head_camera"])
        time.sleep(2.0)
        logger.info("相机流已启动")

    def get_head_frame_rgb(self) -> Optional[np.ndarray]:
        frame = self.robot.camera.get_head_camera_frame()
        if frame is None or frame.get("rgb") is None:
            return None
        img = frame["rgb"]
        if img is None:
            return None

        # If the SDK actually returns BGR but names it rgb, user can set assume_frame_is_rgb=False.
        if self.assume_frame_is_rgb:
            return img
        # treat incoming as BGR -> convert to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def show_preview_loop(
        self,
        window_name: str,
        frame_buffer: Optional[Deque[np.ndarray]] = None,
        overlay_text: Optional[str] = None,
        quit_keys: Tuple[str, ...] = ("q",),
        confirm_keys: Tuple[str, ...] = ("y",),
        waiter: Optional[InputWaiter] = None,
        fps: float = 15.0,
    ) -> str:
        """
        Returns:
          - "confirm" if user confirmed
          - "quit" if user quit
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        target_dt = 1.0 / max(fps, 1e-6)

        while True:
            t0 = time.time()

            img_rgb = self.get_head_frame_rgb()
            if img_rgb is None:
                logger.warning("未获取到相机画面，重试中...")
                time.sleep(0.05)
                continue

            if frame_buffer is not None:
                frame_buffer.append(img_rgb)

            disp = img_rgb.copy()
            if overlay_text:
                cv2.putText(
                    disp, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    disp, "Type 'y' to confirm, Type 'q' to quit（or Type y and press Enter in the terminal）",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # cv2.imshow expects BGR; convert for display only
            disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, disp_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                ch = chr(key).lower()
                if ch in confirm_keys:
                    cv2.destroyWindow(window_name)
                    return "confirm"
                if ch in quit_keys:
                    cv2.destroyWindow(window_name)
                    return "quit"

            if waiter is not None and waiter.is_set():
                waiter.clear()
                cv2.destroyWindow(window_name)
                return "confirm"

            dt = time.time() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

    def cleanup(self):
        logger.info("正在停止相机流...")
        try:
            self.robot.camera.stop_stream()
        except Exception:
            pass
        logger.info("✅ 清理完成")


# ---------------------------
# WebSocket inference
# ---------------------------

async def ws_infer(
    server_url: str,
    obs_frames_rgb: List[np.ndarray],
    goal_rgb: np.ndarray,
    timeout_s: float = 600.0,
    meta: Optional[dict] = None,
) -> Dict:
    assert len(obs_frames_rgb) == 4, "Need exactly 4 observation frames"

    obs_pil = [pil_from_frame_rgb(f) for f in obs_frames_rgb]
    goal_pil = pil_from_frame_rgb(goal_rgb)

    request_data = {
        "type": "inference_request",
        "timestamp": datetime.now().isoformat(),
    }
    if meta:
        request_data["meta"] = meta

    for i, im in enumerate(obs_pil):
        request_data[f"obs_{i}"] = b64_png_from_pil(im)
    request_data["goal"] = b64_png_from_pil(goal_pil)

    t_send = time.time()
    async with websockets.connect(server_url, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request_data))
        try:
            resp = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Server response timeout after {timeout_s}s")
    t_recv = time.time()

    data = json.loads(resp)
    data["_client_latency_s"] = t_recv - t_send
    return data


# ---------------------------
# Trajectory parsing to delta actions
# ---------------------------

def parse_to_delta_actions(response_data: Dict) -> List[Tuple[float, float, float]]:
    """
    Supports:
    - response_data["delta_actions"] = [{"dx":..,"dy":..,"dtheta":..}, ...] or list of triplets
    - response_data["trajectory"] = [{"x":..,"y":..,"yaw":..}, ...]
      -> convert to incremental deltas: dx_i = x_i - x_{i-1}, etc (x_(-1)=0).
    """
    if "delta_actions" in response_data and response_data["delta_actions"] is not None:
        da = response_data["delta_actions"]
        out: List[Tuple[float, float, float]] = []
        for item in da:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                out.append((float(item[0]), float(item[1]), float(item[2])))
            elif isinstance(item, dict):
                out.append((float(item["dx"]), float(item["dy"]), float(item["dtheta"])))
            else:
                raise ValueError(f"Unrecognized delta_actions item: {item}")
        return out

    traj = response_data.get("trajectory")
    if not traj:
        raise ValueError("No delta_actions or trajectory in response")

    # If trajectory already uses dx/dy/dtheta keys
    if isinstance(traj, list) and len(traj) > 0 and isinstance(traj[0], dict) and ("dx" in traj[0] or "dtheta" in traj[0]):
        out = []
        for wp in traj:
            out.append((float(wp.get("dx", 0.0)), float(wp.get("dy", 0.0)), float(wp.get("dtheta", wp.get("yaw", 0.0)))))
        return out

    # Assume cumulative x,y,yaw
    xs = [float(wp["x"]) for wp in traj]
    ys = [float(wp["y"]) for wp in traj]
    yaws = [float(wp.get("yaw", wp.get("theta", 0.0))) for wp in traj]

    out = []
    px, py, pyaw = 0.0, 0.0, 0.0
    for x, y, yaw in zip(xs, ys, yaws):
        out.append((x - px, y - py, normalize_angle(yaw - pyaw)))
        px, py, pyaw = x, y, yaw
    return out


# ---------------------------
# Execution: relative delta navigation + save after each step
# ---------------------------

def execute_delta_actions_with_saving(
    robot: MMK2Robot,
    delta_actions: List[Tuple[float, float, float]],
    io: RobotIO,
    save_dir: Path,
    save_prefix: str = "exec",
) -> Dict:
    """
    执行每个在该段开始机体系下定义的 (dx,dy,dtheta)。
    Decompose: turn(alpha) -> forward(dist) -> turn(beta), where:
      dist = hypot(dx,dy), alpha = atan2(dy,dx), beta = dtheta - alpha
    Save observation frame after each step to save_dir/{save_prefix}_{i:04d}.jpg
    """
    ensure_dir(save_dir)

    start_time = time.time()
    x0, y0, th0 = robot.get_base_pose()
    target_x, target_y, target_th = float(x0), float(y0), float(th0)

    pos_errors: List[float] = []
    yaw_errors: List[float] = []
    step_times: List[float] = []
    step_success: List[bool] = []

    poses_log = []

    for i, (dx, dy, dth) in enumerate(delta_actions):
        step_start = time.time()

        dist = float(np.hypot(dx, dy))
        alpha = float(np.arctan2(dy, dx)) if dist >= 1e-9 else 0.0
        beta = float(dth - alpha)

        logger.info(
            f"→ 步骤 {i}/{len(delta_actions)-1}: Δ=(dx={dx:.3f}, dy={dy:.3f}, dθ={dth:.3f}) "
            f"| α={alpha:.3f}, dist={dist:.3f}, β={beta:.3f}"
        )

        ok = True
        if abs(alpha) > 1e-6:
            ok = ok and (robot.turn_left(alpha, block=True) if alpha > 0 else robot.turn_right(-alpha, block=True))
        if dist > 1e-6:
            ok = ok and robot.move_forward(dist, block=True)
        if abs(beta) > 1e-6:
            ok = ok and (robot.turn_left(beta, block=True) if beta > 0 else robot.turn_right(-beta, block=True))

        step_t = time.time() - step_start
        step_times.append(step_t)
        step_success.append(bool(ok))

        if not ok:
            logger.warning(f"  ⚠️ 步骤 {i} 执行失败（SDK 返回 False）")

        # Update expected absolute target (for rough error stats)
        c, s = float(np.cos(target_th)), float(np.sin(target_th))
        dx_w = dx * c - dy * s
        dy_w = dx * s + dy * c
        target_x += dx_w
        target_y += dy_w
        target_th = normalize_angle(target_th + dth)

        cur_x, cur_y, cur_th = robot.get_base_pose()
        pos_err = float(np.hypot(cur_x - target_x, cur_y - target_y))
        yaw_err = float(abs(angle_diff(cur_th, target_th)))
        pos_errors.append(pos_err)
        yaw_errors.append(yaw_err)

        # Save observation after each step
        frame_rgb = io.get_head_frame_rgb()
        if frame_rgb is not None:
            out_path = save_dir / f"{save_prefix}_{i:04d}.jpg"
            # save as BGR for cv2.imwrite
            cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        else:
            logger.warning("  ⚠️ 无法获取相机画面用于保存该步观测")

        poses_log.append(
            {
                "step": i,
                "ok": bool(ok),
                "time_s": step_t,
                "pose": [float(cur_x), float(cur_y), float(cur_th)],
                "target_pose": [float(target_x), float(target_y), float(target_th)],
                "pos_err_m": pos_err,
                "yaw_err_rad": yaw_err,
                "delta": [float(dx), float(dy), float(dth)],
            }
        )

        logger.info(
            f"  ✓ 步骤 {i} 完成，用时 {step_t:.2f}s | 位置误差={pos_err*1000:.1f}mm | 朝向误差={yaw_err:.3f}rad"
        )

    total_time = time.time() - start_time
    results = {
        "total_time_s": total_time,
        "avg_step_time_s": float(np.mean(step_times)) if step_times else 0.0,
        "success_rate": float(np.mean(step_success)) if step_success else 0.0,
        "avg_pos_err_m": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "avg_yaw_err_rad": float(np.mean(yaw_errors)) if yaw_errors else 0.0,
        "pos_errors_m": pos_errors,
        "yaw_errors_rad": yaw_errors,
        "step_times_s": step_times,
        "step_success": step_success,
    }

    (save_dir / "exec_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (save_dir / "exec_poses.json").write_text(json.dumps(poses_log, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("执行汇总")
    logger.info(f"总耗时： {results['total_time_s']:.2f}s | 成功率={results['success_rate']*100:.1f}%")
    logger.info(f"平均位置误差： {results['avg_pos_err_m']*1000:.1f}mm | 平均朝向误差： {results['avg_yaw_err_rad']:.3f}rad")
    logger.info("=" * 70)

    return results


# ---------------------------
# Goals management
# ---------------------------

def list_goals(goals_dir: Path) -> List[GoalItem]:
    goals_dir = ensure_dir(goals_dir)
    exts = {".png", ".jpg", ".jpeg"}
    items = []
    for p in sorted(goals_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            items.append(GoalItem(name=p.stem, path=p))
    return items


def choose_goal(goals: List[GoalItem]) -> GoalItem:
    if not goals:
        raise RuntimeError("No goals found. Please collect goals first.")
    print("\n可用目标列表：")
    for i, g in enumerate(goals):
        print(f"  [{i}] {g.name} ({g.path.name})")
    while True:
        s = input("请选择目标序号： ").strip()
        if not s.isdigit():
            print("请输入数字。")
            continue
        idx = int(s)
        if 0 <= idx < len(goals):
            return goals[idx]
        print("序号超出范围。")


# ---------------------------
# Main flow
# ---------------------------

async def run_client(args):
    server_url = f"ws://{args.server_host}:{args.server_port}"
    goals_dir = ensure_dir(args.goals_dir)
    run_dir = ensure_dir(Path(args.run_dir) / f"nwm_run_{now_ts()}")
    sent_dir = ensure_dir(run_dir / "sent")
    recv_dir = ensure_dir(run_dir / "received")
    exec_dir = ensure_dir(run_dir / "exec")

    io_robot = RobotIO(
        robot_ip=args.robot_ip,
        camera_sn=args.camera_sn,
        camera_resolution=args.camera_resolution,
        assume_frame_is_rgb=not args.frame_is_bgr,
    )
    io_robot.init_camera()

    waiter = InputWaiter()
    waiter.start()

    try:
        # (1) Goal collection (optional)
        if args.collect_goal:
            overlay = "Goal collection: please push the robot to the target position."
            ret = io_robot.show_preview_loop(
                window_name="goal collect",
                overlay_text=overlay,
                waiter=waiter,
                fps=args.preview_fps,
            )
            if ret != "confirm":
                logger.info("已取消目标采集。")
            else:
                frame_rgb = io_robot.get_head_frame_rgb()
                if frame_rgb is None:
                    raise RuntimeError("无法获取相机画面用于采集目标")
                name = args.goal_name or f"goal_{now_ts()}"
                out_path = goals_dir / f"{name}.png"
                pil = pil_from_frame_rgb(frame_rgb)
                pil.save(out_path)
                logger.info(f"✅ 已保存目标图像： {out_path}")

        # (2) Choose goal
        goals = list_goals(goals_dir)
        goal_item = choose_goal(goals)
        goal_rgb = np.array(Image.open(goal_item.path).convert("RGB"))

        # (3) Move to start: live preview + ring buffer (4 frames)
        frame_buf: Deque[np.ndarray] = deque(maxlen=4)
        overlay = "Start position setup: please push the robot to the start position."
        ret = io_robot.show_preview_loop(
            window_name="Start position setup",
            frame_buffer=frame_buf,
            overlay_text=overlay,
            waiter=waiter,
            fps=args.preview_fps,
        )
        if ret != "confirm":
            logger.info("用户在起点确认前退出。")
            return

        if len(frame_buf) == 0:
            raise RuntimeError("未采集到起点历史观测帧")
        while len(frame_buf) < 4:
            frame_buf.append(frame_buf[-1])

        obs_frames = list(frame_buf)

        # Save sent images locally
        for i, f in enumerate(obs_frames):
            cv2.imwrite(str(sent_dir / f"obs_{i}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sent_dir / "goal.png"), cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2BGR))

        # (4) Inference
        logger.info(f"正在发送推理请求到 {server_url}")
        meta = {"goal_name": goal_item.name, "robot_ip": args.robot_ip}
        t0 = time.time()
        resp = await ws_infer(server_url, obs_frames, goal_rgb, timeout_s=args.timeout_s, meta=meta)
        t1 = time.time()
        latency = resp.get("_client_latency_s", t1 - t0)

        if resp.get("type") != "inference_response" or resp.get("status") != "success":
            logger.error(f"推理失败： {resp}")
            (recv_dir / "error_response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")
            return

        logger.info(f"✅ 已收到推理结果。客户端耗时={latency:.3f}s")
        (recv_dir / "inference_response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")

        delta_actions = parse_to_delta_actions(resp)

        # Save delta actions
        da_path = recv_dir / "delta_actions.txt"
        with da_path.open("w", encoding="utf-8") as f:
            f.write("# dx(m) dy(m) dtheta(rad)\n")
            for dx, dy, dth in delta_actions:
                f.write(f"{dx:.6f} {dy:.6f} {dth:.6f}\n")
        logger.info(f"已保存相对动作序列： {da_path}")

        # (5) Confirm and execute
        print("\n=== 已获取相对动作序列 ===")
        print(f"步数： {len(delta_actions)} | 推理耗时： {latency:.3f}s")
        for i, (dx, dy, dth) in enumerate(delta_actions[:5]):
            print(f"  [{i}] dx={dx:.3f}, dy={dy:.3f}, dθ={dth:.3f} rad")
        if len(delta_actions) > 5:
            print("  ...")

        s = input("\n现在开始执行导航吗？(y/n)： ").strip().lower()
        if s != "y":
            logger.info("用户选择不执行。")
            return

        # Execute and save after each step
        results = execute_delta_actions_with_saving(
            robot=io_robot.robot,
            delta_actions=delta_actions,
            io=io_robot,
            save_dir=exec_dir,
            save_prefix="obs_after",
        )
        (run_dir / "run_summary.json").write_text(json.dumps({"inference_latency_s": latency, **results}, indent=2), encoding="utf-8")
        logger.info(f"✅ 本次运行数据已保存至： {run_dir}")

    finally:
        waiter.stop()
        io_robot.cleanup()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser("NWM 实机推理客户端")

    # Robot
    p.add_argument("--robot-ip", type=str, default="192.168.11.200")
    p.add_argument("--camera-sn", type=str, default="233522073186")
    p.add_argument("--camera-resolution", type=str, default="1280,720,30")
    p.add_argument("--frame-is-bgr", action="store_true",
                   help="如果 SDK 的 frame['rgb'] 实际是 BGR，请设置此项；默认按 RGB 处理")

    # Server
    p.add_argument("--server-host", type=str, required=True)
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--timeout-s", type=float, default=600.0)

    # Storage
    p.add_argument("--goals-dir", type=str, default="./goals")
    p.add_argument("--run-dir", type=str, default="./runs")

    # UI
    p.add_argument("--preview-fps", type=float, default=15.0)

    # Goal collection
    p.add_argument("--collect-goal", action="store_true",
                   help="启用后：启动时先采集一个新的目标图像")
    p.add_argument("--goal-name", type=str, default="",
                   help="采集目标图像的名称（不含扩展名，可选）")

    return p.parse_args()


def main():
    args = parse_args()
    if args.goal_name == "":
        args.goal_name = None
    asyncio.run(run_client(args))


if __name__ == "__main__":
    main()


