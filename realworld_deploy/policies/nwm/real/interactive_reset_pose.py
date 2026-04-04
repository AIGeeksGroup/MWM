"""
交互式初始化脚本：
1) 双臂 reset_to_zero
2) 脊柱移动到“中心高度”
3) 头部看向正前方 (yaw=0, pitch=0)
每一步都需要用户确认后继续执行。

依赖：mmk2_sdk
"""

import sys
import time
import logging
from mmk2_sdk import MMK2Robot, RobotMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("interactive_reset")


# ====== 你可以按实际机器人参数调整这两个范围（来自官方示例的说明） ======
SPINE_HEIGHT_MIN = -0.04
SPINE_HEIGHT_MAX = 0.87

# 头部正前方通常就是 yaw=0, pitch=0（官方示例就是用 set_head(yaw, pitch)）
HEAD_FORWARD_YAW = 0.0
HEAD_FORWARD_PITCH = 0.0


def ask_yes_no(question: str, default_no: bool = True) -> bool:
    """询问用户是否继续。默认更保守：回车=否（防误触）"""
    suffix = " [y/N]" if default_no else " [Y/n]"
    while True:
        try:
            ans = input(question + suffix + ": ").strip().lower()
            if ans == "" and default_no:
                return False
            if ans == "" and not default_no:
                return True
            if ans in ("y", "yes", "是"):
                return True
            if ans in ("n", "no", "否"):
                return False
            print("❌ 请输入 y/n（或 是/否）")
        except KeyboardInterrupt:
            print("\n⚠️  用户中断输入，退出。")
            return False


def get_robot_ip() -> str:
    while True:
        ip = input("请输入机器人IP地址: ").strip()
        if not ip:
            print("❌ IP不能为空")
            continue
        parts = ip.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return ip
        print("❌ IP格式不正确，请重新输入")


def check_and_clear_errors(robot: MMK2Robot) -> bool:
    """执行前后做一次错误检查；有错误就打印并清除。"""
    try:
        if robot.has_errors():
            err = robot.get_last_error()
            summary = robot.get_error_summary()
            logger.error(f"检测到机器人错误: last_error={err}, summary={summary}")
            if ask_yes_no("是否清除错误并继续？", default_no=True):
                robot.clear_errors()
                time.sleep(0.2)
                logger.info("✅ 已清除错误")
                return True
            return False
        return True
    except Exception as e:
        logger.error(f"错误检查失败: {e}")
        return False


def print_joint_snapshot(robot: MMK2Robot):
    """打印关键关节状态（用于执行前确认）"""
    js = robot.get_joint_states()
    if not js or not hasattr(js, "q"):
        logger.warning("无法获取 joint_states")
        return
    q = js.q
    # 官方示例里 q 的切片方式：left(0:6), right(7:13), spine(14:15), head(15:17) 等
    try:
        left = q[0:6]
        right = q[7:13]
        spine = q[14:15]
        head = q[15:17]
        logger.info(f"当前 left_arm  : {left}")
        logger.info(f"当前 right_arm : {right}")
        logger.info(f"当前 spine     : {spine}")
        logger.info(f"当前 head      : {head}")
    except Exception:
        logger.info(f"当前 q: {q}")


def main():
    robot_ip = get_robot_ip()

    # 连接机器人（真实机）
    robot = MMK2Robot(ip=robot_ip, mode=RobotMode.REAL)
    if not robot.is_connected():
        logger.error("机器人连接失败，请检查网络/IP/电源")
        sys.exit(1)

    logger.info("✅ 机器人连接成功")
    print_joint_snapshot(robot)

    # 计算“脊柱中心位置”
    spine_center = (SPINE_HEIGHT_MIN + SPINE_HEIGHT_MAX) / 2.0
    logger.info(f"脊柱中心高度设定为: {spine_center:.3f} (由 [{SPINE_HEIGHT_MIN:.3f}, {SPINE_HEIGHT_MAX:.3f}] 取中点)")

    # 步骤 0：执行前错误检查
    if not check_and_clear_errors(robot):
        sys.exit(1)

    # 步骤 1：双臂重置到零点
    logger.info("STEP 1/3: 将双臂重置为零点姿态 (reset_to_zero('all'))")
    print_joint_snapshot(robot)
    if not ask_yes_no("确认执行 STEP 1？", default_no=True):
        logger.info("用户取消，退出。")
        return

    try:
        robot.reset_to_zero("all")  # 官方示例里用过
        logger.info("✅ STEP 1 已发送/执行完成")
    except Exception as e:
        logger.error(f"STEP 1 执行失败: {e}")
        return

    if not check_and_clear_errors(robot):
        return

    # 步骤 2：脊柱降到中心位置
    logger.info("STEP 2/3: 将脊柱移动到中心高度 (set_spine(center))")
    print_joint_snapshot(robot)
    if not ask_yes_no(f"确认执行 STEP 2？目标 spine={spine_center:.3f}", default_no=True):
        logger.info("用户取消，退出。")
        return

    try:
        ok = robot.set_spine(spine_center, block=True)  # 官方示例里用过
        logger.info(f"✅ STEP 2 执行结果: {'成功' if ok else '失败'}")
    except Exception as e:
        logger.error(f"STEP 2 执行失败: {e}")
        return

    if not check_and_clear_errors(robot):
        return

    # 步骤 3：头部看向正前方
    logger.info("STEP 3/3: 头部看向正前方 (set_head(yaw=0, pitch=0))")
    print_joint_snapshot(robot)
    if not ask_yes_no(f"确认执行 STEP 3？目标 head=({HEAD_FORWARD_YAW:.2f}, {HEAD_FORWARD_PITCH:.2f})", default_no=True):
        logger.info("用户取消，退出。")
        return

    try:
        ok = robot.set_head(HEAD_FORWARD_YAW, HEAD_FORWARD_PITCH, block=True)  # 官方示例里用过
        logger.info(f"✅ STEP 3 执行结果: {'成功' if ok else '失败'}")
    except Exception as e:
        logger.error(f"STEP 3 执行失败: {e}")
        return

    if not check_and_clear_errors(robot):
        return

    logger.info("🎉 全部步骤完成")
    print_joint_snapshot(robot)


if __name__ == "__main__":
    main()
