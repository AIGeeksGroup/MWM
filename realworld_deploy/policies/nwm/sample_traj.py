"""
OMPL (Python bindings) + PRM/PRM*：
- 输入：多个障碍物多边形（每个多边形由 (x,y) 点按顺/逆时针给出）
- 输出：随机采样多条“无碰撞轨迹”(start->goal)，碰撞依据：轨迹折线段是否与任一多边形相交，
        且轨迹点不允许落在多边形内部（可按需调整允许贴边策略）。

依赖：
  pip install shapely

说明：
- 用 SE2StateSpace，但只用 x,y（yaw 固定为 0），方便以后扩展到带朝向。
- 运动有效性：用 shapely 判断线段是否与多边形相交/穿过。
"""

import random
from typing import List, Tuple, Optional

from ompl import base as ob
from ompl import geometric as og

from shapely.geometry import Polygon, LineString, Point
from shapely.prepared import prep

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
import json
import os
import math

output_dir = "/root/autodl-tmp/embodiedAI/DISCOVERSE/policies/nwm/data_collect_output/prm"
os.makedirs(output_dir, exist_ok=True)

def visualize_bounds_obstacles_and_trajs(bounds, obstacles, trajs):
    """
    bounds: (xmin, xmax, ymin, ymax)
    obstacles: List[List[(x,y)]]
    trajs: List[List[(x,y)]]
    """
    xmin, xmax, ymin, ymax = bounds

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot bounds
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        'k--',
        linewidth=1.5,
        label="Bounds"
    )

    # plot obstacles
    patches = []
    for poly in obstacles:
        patches.append(MplPolygon(poly, closed=True))

    obstacle_collection = PatchCollection(
        patches,
        facecolor='gray',
        edgecolor='black',
        alpha=0.6,
        label="Obstacles"
    )
    ax.add_collection(obstacle_collection)

    # plot traj
    for i, traj in enumerate(trajs):
        traj = np.asarray(traj)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=1.2,
            alpha=0.8
        )
        # 起点 / 终点
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=20)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=20)

    ax.set_aspect('equal')
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("PRM Sampled Collision-Free Trajectories")

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(os.path.join(output_dir, "traj_prm.png"))

def build_obstacles(polygons_xy: List[List[Tuple[float, float]]]):
    polys = []
    for pts in polygons_xy:
        poly = Polygon(pts)
        if not poly.is_valid:
            # 如果点序/自交导致无效，尝试修复
            poly = poly.buffer(0)
        polys.append(poly)
    # prepared geometry，加速 contains/intersects
    prepared = [prep(p) for p in polys]
    return polys, prepared


class PolygonStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si: ob.SpaceInformation, prepared_polys):
        super().__init__(si)
        self.prepared_polys = prepared_polys

    def isValid(self, state) -> bool:
        x = state.getX()
        y = state.getY()
        pt = Point(x, y)
        # 禁止点在障碍物内部（不含边界）
        for p in self.prepared_polys:
            if p.contains(pt):
                return False
        return True

# ----------------------------
# 运动有效性：轨迹线段不能与任何多边形相交/穿过
#    这里把“相交”定义为：线段与多边形边界相交 或 线段有一段在多边形内部
# ----------------------------
class PolygonMotionValidator(ob.MotionValidator):
    def __init__(self, si: ob.SpaceInformation, polys, prepared_polys):
        super().__init__(si)
        self.polys = polys
        self.prepared_polys = prepared_polys
        self.si = si

    def checkMotion(self, s1, s2, lastValid=None) -> bool:
        x1, y1 = s1.getX(), s1.getY()
        x2, y2 = s2.getX(), s2.getY()
        seg = LineString([(x1, y1), (x2, y2)])

        for pprep, poly in zip(self.prepared_polys, self.polys):
            # 与多边形相交（含穿过/触碰边界/重合）
            if pprep.intersects(seg):
                if seg.touches(poly):
                    return False
                return False

            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            if pprep.contains(Point(mx, my)):
                return False

        # 也确保端点有效（通常由 StateValidityChecker + discretization 也能保证）
        return self.si.isValid(s1) and self.si.isValid(s2)

def sample_free_state(space: ob.SE2StateSpace, si: ob.SpaceInformation,
                      xmin, xmax, ymin, ymax, max_tries=10000) -> ob.State:
    for _ in range(max_tries):
        st = ob.State(space)
        st().setX(random.uniform(xmin, xmax))
        st().setY(random.uniform(ymin, ymax))
        st().setYaw(0.0)
        if si.isValid(st()):
            return st
    raise RuntimeError("Failed to sample a free state. Check bounds/obstacles.")

def path_to_xy(path: og.PathGeometric) -> List[Tuple[float, float]]:
    pts = []
    for i in range(path.getStateCount()):
        s = path.getState(i)
        pts.append((s.getX(), s.getY()))
    return pts

def sample_collision_free_trajectories_with_prm(
    polygons_xy: List[List[Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],     # (xmin, xmax, ymin, ymax)
    n_paths: int = 20,
    planner_type: str = "PRMstar",                # "PRM" or "PRMstar"
    build_roadmap_time: float = 1.0,
    solve_time_per_query: float = 0.5,
    interpolate_n: int = 200,                     # 轨迹插值点数（用于更密集输出）
    seed: Optional[int] = 0,
) -> List[List[Tuple[float, float]]]:

    if seed is not None:
        random.seed(seed)

    xmin, xmax, ymin, ymax = bounds
    polys, prepared = build_obstacles(polygons_xy)

    # 1) 定义状态空间
    space = ob.SE2StateSpace()
    sb = ob.RealVectorBounds(2)
    sb.setLow(0, xmin); sb.setHigh(0, xmax)
    sb.setLow(1, ymin); sb.setHigh(1, ymax)
    space.setBounds(sb)

    # 2) SimpleSetup
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    # 3) Validity + MotionValidator
    svc = PolygonStateValidityChecker(si, prepared)
    ss.setStateValidityChecker(svc)

    mv = PolygonMotionValidator(si, polys, prepared)
    si.setMotionValidator(mv)

    # 4) PRM / PRMstar
    if planner_type.lower() == "prmstar":
        planner = og.PRMstar(si)
    else:
        planner = og.PRM(si)
    ss.setPlanner(planner)

    # 5) 可选：设置离散检查分辨率（作为兜底）
    #    值越小检查越密，但更慢；仅靠 MotionValidator 已经是“线段-多边形”精确判断
    si.setStateValidityCheckingResolution(0.005)  # 相对空间跨度的比例

    ss.setup()

    # 6) 预先建路网（让后续多次 query 更快）
    try:
        planner.growRoadmap(ob.timedPlannerTerminationCondition(build_roadmap_time))
    except Exception:
        pass

    trajectories = []
    num_tries = 0
    max_total_tries = n_paths * 50

    while len(trajectories) < n_paths and num_tries < max_total_tries:
        num_tries += 1

        start = ob.State(space)
        start.random()
        goal = ob.State(space)
        goal.random()

        ss.setStartAndGoalStates(start, goal)

        solved = ss.solve(solve_time_per_query)
        if solved:
            path = ss.getSolutionPath()
            if interpolate_n and interpolate_n > 2:
                path.interpolate(interpolate_n)
            trajectories.append(path_to_xy(path))

        try:
            ss.getPlanner().clearQuery()
        except Exception:
            pass
        ss.clearStartStates()

    if len(trajectories) < n_paths:
        print(f"[WARN] Only found {len(trajectories)}/{n_paths} paths. "
              f"Try increasing build_roadmap_time/solve_time_per_query or enlarging bounds.")

    return trajectories

def compute_yaw_from_trajs(data):
    """
    data: List[List[(x, y)]]
    return: List[List[yaw]]  # yaw in radians
    """
    all_yaws = []

    for traj in data:
        n = len(traj)
        yaws = []

        for i in range(n - 1):
            x0, y0 = traj[i]
            x1, y1 = traj[i + 1]
            yaw = math.atan2(y1 - y0, x1 - x0)  # radians
            yaws.append(yaw)

        yaws.append(random.uniform(-math.pi, math.pi))

        all_yaws.append(yaws)

    return all_yaws


if __name__ == "__main__":
    obstacles = [
        [
            (0, 0.8),
            (0.12, -1.33),
            (7.88, -1.34),
            (7.88, 0.8)
        ],
    ]
    # 空间边界
    bounds = (-2.22, 17.60, -3.37, 0.8)

    trajs = sample_collision_free_trajectories_with_prm(
        polygons_xy=obstacles,
        bounds=bounds,
        n_paths=200,
        planner_type="PRMstar",
        build_roadmap_time=2.0,
        solve_time_per_query=0.8,
        interpolate_n=30,
        seed=42,
    )

    print("Trajectories:", len(trajs))
    print("First trajectory length:", len(trajs[0]) if trajs else 0)

    visualize_bounds_obstacles_and_trajs(bounds, obstacles, trajs)

    # calculate yaws
    all_yaws = compute_yaw_from_trajs(trajs)

    # save
    data_json = []
    for i, traj in enumerate(trajs):
        data_json.append({"positions": [[float(x), float(y)] for x, y in traj], "yaws": all_yaws[i]})
    with open(os.path.join(output_dir, "traj_prm.json"), "w") as f:
        json.dump(data_json, f)

