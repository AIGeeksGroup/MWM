"""
底盘航点导航测试 - 三种控制模式
"""

import time
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from mmk2_sdk import MMK2Robot, RobotMode


class BaseNavigationTester:
    """底盘导航测试类"""
    
    def __init__(self, robot: MMK2Robot):
        self.robot = robot
        self.waypoint_distance = 0.5  # 航点间距(米)
        self.num_waypoints = 8
        self.total_distance = (self.num_waypoints - 1) * self.waypoint_distance
        self.control_orientation = False

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """
        计算两个角度之间的最小差值（弧度），结果范围为 [-pi, pi]。
        这在判断 theta 是否到位时比直接相减更稳健。
        """
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return d
        
    def set_orientation_control(self, enable: bool):
        """设置是否控制朝向"""
        self.control_orientation = enable
        logger.info(f"朝向控制: {'开启' if enable else '关闭'}")

    def generate_delta_actions(self) -> List[Tuple[float, float, float]]:
        """生成测试相对位移序列 (dx, dy, dtheta[rad])"""
        logger.info("\n" + "="*70)
        logger.info("生成测试相对位移序列")
        logger.info("="*70)

        current_pose = self.robot.get_base_pose()
        logger.info(
            f"起始位姿: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, theta={current_pose[2]:.3f}"
        )

        # 约定：沿机器人“前向”走直线，每步走 waypoint_distance
        # 你也可以把 step_distance 单独设成 self.step_distance
        step_distance = float(self.waypoint_distance)
        num_steps = int(self.num_waypoints)

        # 可选：每隔 k 步转一次，形成简单“折线/蛇形”轨迹
        # 如果不需要转向，把 turn_every_k 设为 0 或 None 即可
        turn_every_k = getattr(self, "turn_every_k", 1)      # e.g., 5
        turn_angle = getattr(self, "turn_angle", np.pi / 4)        # rad, e.g., np.deg2rad(10)

        delta_actions: List[Tuple[float, float, float]] = []

        for i in range(num_steps):
            dx = step_distance
            dy = 0.0

            if self.control_orientation:
                # 保持朝向不变：不转
                dtheta = 0.0
            else:
                # 默认也不转（否则 mode4 需要数值，不能用 None）
                dtheta = 0.0

                # 如果你希望“自由朝向”体现为周期性转向（蛇形），可启用这段
                if turn_every_k and turn_every_k > 0 and (i + 1) % turn_every_k == 0:
                    # 交替左/右转
                    sign = 1.0 if ((i // turn_every_k) % 2 == 0) else -1.0
                    dtheta = float(sign * turn_angle)

            delta_actions.append((dx, dy, dtheta))

        total_dist = step_distance * num_steps
        logger.info(f"生成 {len(delta_actions)} 段相对位移，总前进距离: {total_dist:.2f}m")
        if turn_every_k and turn_every_k > 0 and not self.control_orientation:
            logger.info(f"转向设置: 每 {turn_every_k} 步转一次，角度 {turn_angle:.3f} rad（左右交替）")

        for i, (dx, dy, dth) in enumerate(delta_actions):
            logger.info(f"  Δ{i}: dx={dx:.3f}, dy={dy:.3f}, dtheta={dth:.3f} rad")

        return delta_actions

        
    def generate_waypoints(self) -> List[Tuple[float, float, float]]:
        """生成测试航点序列"""
        logger.info("\n" + "="*70)
        logger.info("生成测试航点")
        logger.info("="*70)
        
        current_pose = self.robot.get_base_pose()
        logger.info(f"起始位姿: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, theta={current_pose[2]:.3f}")
        
        waypoints = []
        if self.control_orientation:
            fixed_theta = current_pose[2]
            for i in range(self.num_waypoints):
                x = current_pose[0] + i * self.waypoint_distance
                y = current_pose[1]
                waypoints.append((x, y, fixed_theta))
        else:
            for i in range(self.num_waypoints):
                x = current_pose[0] + i * self.waypoint_distance
                y = current_pose[1]
                waypoints.append((x, y, None))
            
        logger.info(f"生成 {len(waypoints)} 个航点，总距离: {self.total_distance:.2f}m")
        for i, wp in enumerate(waypoints):
            if wp[2] is not None:
                logger.info(f"  航点 {i}: x={wp[0]:.3f}, y={wp[1]:.3f}, theta={wp[2]:.3f}")
            else:
                logger.info(f"  航点 {i}: x={wp[0]:.3f}, y={wp[1]:.3f}, theta=自由")
        
        return waypoints
    
    def mode1_blocking_navigation(self, waypoints: List[Tuple[float, float, float]]) -> dict:
        """模式1: 阻塞式逐点导航"""
        logger.info("\n" + "="*70)
        logger.info("模式1: 阻塞式逐点导航")
        logger.info("="*70)
        
        start_time = time.time()
        errors = []
        
        for i, (x, y, theta) in enumerate(waypoints):
            waypoint_start = time.time()
            
            if theta is None:
                current_theta = self.robot.get_base_pose()[2]
                logger.info(f"→ 航点 {i}/{len(waypoints)-1}: ({x:.3f}, {y:.3f}, 当前朝向)")
                success = self.robot.move_base(x, y, current_theta, block=True)
            else:
                logger.info(f"→ 航点 {i}/{len(waypoints)-1}: ({x:.3f}, {y:.3f}, {theta:.3f})")
                success = self.robot.move_base(x, y, theta, block=True)
            
            waypoint_time = time.time() - waypoint_start
            
            if not success:
                logger.warning(f"  ⚠️  航点 {i} 导航失败")
            
            current_pose = self.robot.get_base_pose()
            error = np.sqrt((current_pose[0] - x)**2 + (current_pose[1] - y)**2)
            errors.append(error)
            
            logger.info(f"  ✓ 到达航点 {i}, 耗时: {waypoint_time:.2f}s, 误差: {error*1000:.1f}mm")
        
        total_time = time.time() - start_time
        
        results = {
            "method": "阻塞式逐点导航",
            "total_time": total_time,
            "avg_error": np.mean(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "errors": errors
        }
        
        self._print_summary(results)
        return results
    
    def mode2_nonblocking_navigation(self, waypoints: List[Tuple[float, float, float]]) -> dict:
        """模式2: 非阻塞式逐点导航"""
        logger.info("\n" + "="*70)
        logger.info("模式2: 非阻塞式逐点导航")
        logger.info("="*70)
        
        start_time = time.time()
        errors = []
        
        position_tolerance = 0.03
        angle_tolerance = 0.1
        check_interval = 0.05
        timeout_per_waypoint = 10.0
        
        for i, (x, y, theta) in enumerate(waypoints):
            if theta is None:
                current_theta = self.robot.get_base_pose()[2]
                target_theta = current_theta
                logger.info(f"→ 航点 {i}/{len(waypoints)-1}: ({x:.3f}, {y:.3f}, 当前朝向)")
            else:
                target_theta = theta
                logger.info(f"→ 航点 {i}/{len(waypoints)-1}: ({x:.3f}, {y:.3f}, {theta:.3f})")
            
            waypoint_start = time.time()
            success = self.robot.move_base(x, y, target_theta, block=False)
            
            if not success:
                logger.warning(f"  ⚠️  航点 {i} 命令发送失败")
                continue
            
            reached = False
            while time.time() - waypoint_start < timeout_per_waypoint:
                current_pose = self.robot.get_base_pose()
                pos_error = np.sqrt((current_pose[0] - x)**2 + (current_pose[1] - y)**2)
                
                if theta is not None:
                    angle_error = abs(self._angle_diff(current_pose[2], target_theta))
                    if pos_error < position_tolerance and angle_error < angle_tolerance:
                        reached = True
                        break
                else:
                    if pos_error < position_tolerance:
                        reached = True
                        break
                
                time.sleep(check_interval)
            
            waypoint_time = time.time() - waypoint_start
            current_pose = self.robot.get_base_pose()
            final_error = np.sqrt((current_pose[0] - x)**2 + (current_pose[1] - y)**2)
            errors.append(final_error)
            
            status = "✓ 到达" if reached else "⚠️  超时"
            logger.info(f"  {status} 航点 {i}, 耗时: {waypoint_time:.2f}s, 误差: {final_error*1000:.1f}mm")
        
        total_time = time.time() - start_time
        
        results = {
            "method": "非阻塞式逐点导航",
            "total_time": total_time,
            "avg_error": np.mean(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "errors": errors
        }
        
        self._print_summary(results)
        return results
    
    # ✅ 用 move_forward 连续前进 7 段 0.25m
    def mode3_forward_segments(self) -> dict:
        """
        模式3: 连续前进 7×0.25m（使用 move_forward）
        不再使用底层速度接口 _control_velocity
        """
        logger.info("\n" + "="*70)
        logger.info("模式3: 连续前进 7×0.25m（move_forward）")
        logger.info("="*70)

        start_time = time.time()
        segment_len = self.waypoint_distance  # 0.25m
        num_segments = self.num_waypoints - 1  # 7 段

        errors = []

        # 记录起点
        start_pose = self.robot.get_base_pose()
        logger.info(f"起始位姿: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, theta={start_pose[2]:.3f}")
        logger.info("说明: 误差评估假定理想轨迹沿世界坐标系 +X 方向直线，仅用于近似对比。")

        for i in range(num_segments):
            seg_start = time.time()
            logger.info(f"→ 正在前进第 {i+1}/{num_segments} 段: {segment_len:.3f} m")

            success = self.robot.move_forward(segment_len, block=True)
            seg_time = time.time() - seg_start

            if not success:
                logger.warning(f"⚠️ 第 {i+1} 段 move_forward 失败")
                # 失败也继续往下走，记录一下误差
            current_pose = self.robot.get_base_pose()

            # 理论目标：x 方向前进 i+1 段，y 不变（只做近似评估）
            target_x = start_pose[0] + (i + 1) * segment_len
            target_y = start_pose[1]
            err = np.hypot(current_pose[0] - target_x, current_pose[1] - target_y)
            errors.append(err)

            logger.info(
                f"  段 {i+1} 完成, 耗时: {seg_time:.2f}s, 误差: {err*1000:.1f}mm "
                f"(x={current_pose[0]:.3f}, y={current_pose[1]:.3f})"
            )

        total_time = time.time() - start_time
        final_pose = self.robot.get_base_pose()
        final_error = np.hypot(
            final_pose[0] - (start_pose[0] + num_segments * segment_len),
            final_pose[1] - start_pose[1],
        )

        results = {
            "method": "连续前进 7×0.25m（move_forward）",
            "total_time": total_time,
            "avg_error": np.mean(errors) if errors else final_error,
            "max_error": np.max(errors) if errors else final_error,
            "min_error": np.min(errors) if errors else final_error,
            "final_error": final_error,
            "errors": errors,
        }
        logger.info(f"\n最终位置误差: {final_error*1000:.1f}mm")
        self._print_summary(results)
        return results
    
    def mode4_blocking_navigation(self, delta_actions: List[Tuple[float, float, float]]) -> dict:
        """
        模式4: 相对位移导航（阻塞式）
        输入 delta_actions: 多段相对位移 (dx, dy, dtheta[rad])，在每段开始时的机器人坐标系下定义。
        通过 SDK 的 move_forward / turn_left / turn_right 执行。
        """
        def _angle_diff(a: float, b: float) -> float:
            """返回 a-b 的最小角差（绝对值可做误差）"""
            return _normalize_angle(a - b)
        
        def _normalize_angle(angle: float) -> float:
            """将角度归一化到 [-pi, pi)"""
            return (angle + np.pi) % (2 * np.pi) - np.pi
    
        logger.info("\n" + "=" * 70)
        logger.info("模式4: 相对位移导航（阻塞式）")
        logger.info("=" * 70)

        start_time = time.time()

        # 以起点为基准，累计期望的绝对目标（用于误差统计）
        x0, y0, th0 = self.robot.get_base_pose()
        target_x, target_y, target_th = float(x0), float(y0), float(th0)

        pos_errors = []
        yaw_errors = []
        step_times = []
        step_success = []

        for i, (dx, dy, dth) in enumerate(delta_actions):
            step_start = time.time()

            # 计算该段的执行分解：turn(alpha) -> forward(dist) -> turn(beta)
            dist = float(np.hypot(dx, dy))
            if dist < 1e-9:
                alpha = 0.0
            else:
                alpha = float(np.arctan2(dy, dx))
            beta = float(dth - alpha)

            logger.info(
                f"→ Step {i}/{len(delta_actions)-1}: "
                f"Δ=(dx={dx:.3f}, dy={dy:.3f}, dθ={dth:.3f}rad) | "
                f"decompose: α={alpha:.3f}, dist={dist:.3f}, β={beta:.3f}"
            )

            ok = True

            # 1) turn alpha
            if abs(alpha) > 1e-6:
                if alpha > 0:
                    ok = ok and self.robot.turn_left(alpha, block=True)
                else:
                    ok = ok and self.robot.turn_right(-alpha, block=True)

            # 2) move forward dist
            if dist > 1e-6:
                ok = ok and self.robot.move_forward(dist, block=True)

            # 3) turn beta (使最终朝向变化为 dtheta)
            if abs(beta) > 1e-6:
                if beta > 0:
                    ok = ok and self.robot.turn_left(beta, block=True)
                else:
                    ok = ok and self.robot.turn_right(-beta, block=True)

            step_t = time.time() - step_start
            step_times.append(step_t)
            step_success.append(bool(ok))

            if not ok:
                logger.warning(f"  ⚠️  Step {i} 执行失败（SDK 返回 False）")

            # === 更新期望目标（绝对坐标），用于误差统计 ===
            # delta 是“该段开始时机体系”，需要用 target_th（该段开始时的期望朝向）旋转到世界系
            c, s = float(np.cos(target_th)), float(np.sin(target_th))
            dx_w = dx * c - dy * s
            dy_w = dx * s + dy * c
            target_x += dx_w
            target_y += dy_w
            target_th = _normalize_angle(target_th + dth)

            # === 读实际位姿并算误差 ===
            cur_x, cur_y, cur_th = self.robot.get_base_pose()
            pos_err = float(np.hypot(cur_x - target_x, cur_y - target_y))
            yaw_err = float(abs(_angle_diff(cur_th, target_th)))

            pos_errors.append(pos_err)
            yaw_errors.append(yaw_err)

            logger.info(
                f"  ✓ Step {i} done, 耗时: {step_t:.2f}s, "
                f"位置误差: {pos_err*1000:.1f}mm, 朝向误差: {yaw_err:.3f}rad"
            )

        total_time = time.time() - start_time

        results = {
            "method": "相对位移导航（阻塞式）",
            "total_time": total_time,
            "avg_step_time": float(np.mean(step_times)) if step_times else 0.0,
            "success_rate": float(np.mean(step_success)) if step_success else 0.0,

            "avg_pos_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
            "max_pos_error": float(np.max(pos_errors)) if pos_errors else 0.0,
            "min_pos_error": float(np.min(pos_errors)) if pos_errors else 0.0,

            "avg_yaw_error": float(np.mean(yaw_errors)) if yaw_errors else 0.0,
            "max_yaw_error": float(np.max(yaw_errors)) if yaw_errors else 0.0,
            "min_yaw_error": float(np.min(yaw_errors)) if yaw_errors else 0.0,

            "pos_errors": pos_errors,
            "yaw_errors": yaw_errors,
            "step_times": step_times,
            "step_success": step_success,
        }

        self._print_summary(results)
        return results
    
    def _print_summary(self, result: dict):
        """打印测试结果摘要"""
        logger.info("\n" + "-"*70)
        logger.info("测试结果摘要")
        logger.info("-"*70)
        logger.info(f"控制模式: {result['method']}")
        logger.info(f"总耗时:   {result['total_time']:.2f} 秒")
        logger.info(f"平均误差: {result['avg_error']*1000:.1f} mm")
        logger.info(f"最大误差: {result['max_error']*1000:.1f} mm")
        logger.info(f"最小误差: {result['min_error']*1000:.1f} mm")
        if 'final_error' in result:
            logger.info(f"终点误差: {result['final_error']*1000:.1f} mm")
        logger.info(f"平均速度: {self.total_distance/result['total_time']:.3f} m/s")
        logger.info("="*70)


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("底盘航点平滑导航测试程序 (基于SDK)")
    print("测试配置: 8个航点，间距0.25m，总距离1.75m")
    print("="*70)
    
    robot_ip = input("\n请输入机器人IP地址: ").strip()
    if not robot_ip:
        logger.error("❌ IP地址不能为空")
        return
    
    logger.info(f"\n正在连接机器人: {robot_ip}")
    robot = MMK2Robot(ip=robot_ip, mode=RobotMode.REAL)
    
    if not robot.is_connected():
        logger.error("❌ 机器人连接失败")
        return
    
    logger.info("✅ 机器人连接成功")
    
    tester = BaseNavigationTester(robot)
    
    while True:
        orientation_status = "开启" if tester.control_orientation else "关闭 (推荐)"
        print("\n" + "="*70)
        print("底盘航点导航测试")
        print("="*70)
        print("\n1. 阻塞式逐点导航")
        print("2. 非阻塞式逐点导航")
        print("3. 连续前进 7×0.25m （move_forward）")
        print(f"4. 设置朝向控制 (当前: {orientation_status})")
        print("5. 相对位移导航（阻塞式）")
        print("0. 退出程序")
        print("="*70)
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "0":
            logger.info("👋 退出程序")
            break
        
        if choice == "4":
            print("\n1. 开启朝向控制")
            print("2. 关闭朝向控制 (推荐)")
            orient_choice = input("请选择 (1-2): ").strip()
            
            if orient_choice == "1":
                tester.set_orientation_control(True)
            elif orient_choice == "2":
                tester.set_orientation_control(False)
            else:
                logger.warning("无效选择")
            continue
        
        if choice not in ["1", "2", "3", "5"]:
            logger.warning("❌ 无效选择")
            continue
        
        logger.info("\n📍 获取当前位姿...")
        current_pose = robot.get_base_pose()
        logger.info(f"当前位姿: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, theta={current_pose[2]:.3f}")
        
        if choice in ["1", "2", "3"]:
            waypoints = tester.generate_waypoints()
        elif choice in ["5"]:
            waypoints = tester.generate_delta_actions()
        
        print("\n⚠️  安全提示:")
        print(f"   - 机器人将向前移动约 {tester.total_distance:.2f}m")
        print("   - 请确保前方路径无障碍物")
        
        confirm = input("\n确认开始测试? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("取消测试")
            continue
        
        try:
            if choice == "1":
                result = tester.mode1_blocking_navigation(waypoints)
            elif choice == "2":
                result = tester.mode2_nonblocking_navigation(waypoints)
            elif choice == "3":
                # ✅ 不再用 _control_velocity
                result = tester.mode3_forward_segments()
            elif choice == "5":
                result = tester.mode4_blocking_navigation(waypoints)
            
            logger.info("\n✅ 测试完成!")
            
            print("\n1. 自动返回起点")
            print("2. 手动推回")
            print("3. 不返回")
            
            return_choice = input("\n请选择 (1-3): ").strip()
            
            if return_choice == "1":
                logger.info(f"🔙 正在返回起点，后退 {tester.total_distance:.2f}m...")
                robot.move_backward(tester.total_distance, block=True)
                time.sleep(0.5)
                
                final_pose = robot.get_base_pose()
                return_error = np.sqrt(
                    (final_pose[0] - current_pose[0])**2 + 
                    (final_pose[1] - current_pose[1])**2
                )
                logger.info(f"✓ 已返回起点，误差: {return_error*1000:.1f}mm")
                
            elif return_choice == "2":
                logger.info("👋 请手动将机器人推回起点")
                logger.info(f"   目标位姿: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}")
                input("\n推回完成后按回车键继续...")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  测试被用户中断")
        except Exception as e:
            logger.error(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
    #test()