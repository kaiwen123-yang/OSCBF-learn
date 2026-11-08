# Piper H / OSCBF

2026-11-08 最新状态：完整执行器已提供 `baseline/cbf/oscbf` 方法入口，但无障碍完整OSCBF尚未完成实机验证。最新准备动作因实时时钟跳变造成旧反馈超时误判而急停，已单独清急停并保持失能；详细原因与当前代码清理见 [最新交接](PROJECT_STATE.md) 和 [执行流程](docs/EXECUTION_POLICY.md)。


AgileX Piper H 实机项目，Ubuntu 22.04 + ROS 2 Humble + Python 3.10，标准夹爪，无项目相机。

**已有成果：固定位置完整抓放实机基线，以及独立的 OSCBF/普通 CBF 虚拟平面实机对比。**
完整抓放尚未接入 OSCBF，也没有完成已知实体障碍绕行。失败记录和未解决问题保留在仓库中。

先读 [PROJECT_STATE.md](PROJECT_STATE.md) 的最新交接，再看 [阶段索引](docs/STAGES.md) 和 [换机说明](docs/MIGRATION.md)。

## 当前阶段

| 阶段 | 状态 | 证据与缺口 |
|---|---|---|
| A 基础实机运动 | 已完成 | 实测 CAN 关节状态、MOVE_J、CPV 小幅运动记录在 logs/ |
| B MoveIt 完整抓放基线 | 部分完成 | 多次无额外障碍连续抓放成功；缺已知实体障碍绕行 |
| C 论文仿真与 Piper 模型迁移 | 已完成 | 官方 Panda 示例和 Piper 模型仿真分别记录；不能当作真机成果 |
| D OSCBF 实机控制 | 已完成 | 实测反馈→QP→CPV；同参考平面对比已运行，有小幅负 h，保留原值 |
| E 完整任务接入 OSCBF | 尚未完成 | 当前抓放是 MoveIt 路径＋关节速度跟踪 |
| F 结果与演示 | 部分完成 | 原始数据、图、复跑入口齐备；尚无仓库内可确认的连续视频和完整任务对比 |

## 新设备开始

```bash
git clone --recurse-submodules https://github.com/kaiwen123-yang/OSCBF-learn.git
cd OSCBF-learn
```

Ubuntu 22.04 安装基础工具后，建立项目自己的 Python 环境：

```bash
sudo apt-get update
sudo apt-get install git python3.10-venv build-essential
bash scripts/bootstrap_new_device.sh
```

需要重新运行 MoveIt 规划时，先安装 [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)，然后：

```bash
bash scripts/bootstrap_new_device.sh --with-ros
```

以上只安装软件和运行离线检查，不连接 CAN。`envs/` 和 ROS 编译目录由新设备重建，依赖版本在 [requirements/](requirements/)。

先验证已有计划与分析，无硬件动作：

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py results/pick_place_plan_20261104_224921_126900/task.json
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/analyze_piper_diagnostics.py
```

## 主要证据

- [固定共同参考 OSCBF/CBF 实机平面对比](results/plane_pair_20261104_223600_152131/summary.json)：激活窗口切向 RMS 为 0.181/6.204 mm；最小 h 为 -0.645/-0.605 mm，不能宣称严格全程不越界。
- [首次完整慢速抓放](results/pick_place_hardware_20261104_230919_470403/summary.json)和[慢速复跑](results/pick_place_hardware_20261104_235251_893243/summary.json)：用户现场确认成功。
- [1.5 倍抓放](results/pick_place_hardware_20261106_220347_670955/summary.json)：用户确认成功，55.8 秒。
- [两类故障复跑与统一分析](results/pick_place_hardware_20261106_220347_670955/ANALYSIS.md)：两次抓放、两次 MOVE_J 零位保持；旧故障未复现，捕获 CPV 段末 CAN 错误。
- [诊断对比图](results/pick_place_hardware_20261106_220347_670955/diagnostic_comparison.png)。

## 目录

| 路径 | 内容 |
|---|---|
| scripts/、piper_oscbf/ | 安装、模型、规划、控制、记录和离线分析代码 |
| config/、models/ | 场景、限速、虚拟平面、Piper H 模型 |
| results/ | 全部现存仿真、规划、实机运行及逐次原始数据，包括失败包 |
| logs/、log/ | 设备状态、初始化、恢复、环境、ROS 运行记录 |
| src/oscbf、src/pyAgxArm | 固定提交的论文实现及厂家 SDK 子模块 |
| ros_ws/src/agx_arm_ros | 固定提交的厂家 ROS 驱动，递归包含 URDF/网格子模块 |
| docs/ | 阶段证据、换机说明和操作入口 |

## 接着开发什么

首要问题是 **CPV 段末零速指令紧邻上一组控制指令，伴随 CAN 错误突发**。待验证统一发送节拍能否消除异常。
J2 曾返回 `0xA4`，根因未确认；今天的近零位保持成功不替代原前置状态下的完整回零复现。

硬件入口见 [PICK_PLACE.md](docs/PICK_PLACE.md)、[HARDWARE_PAIR.md](docs/HARDWARE_PAIR.md)。历史文件中的姿态、使能状态和绝对路径只代表采集当时，不代表新设备当前状态。实机动作由现场操作者启动，同一时刻仅保留一个命令发送者。

本仓库于 2026-11-07 从现有项目按六组阶段内容迁入；提交日期为迁移日期，不伪造此前开发历史。原始运行时间保留在文件和数据中。

上游：论文 [arXiv:2503.06736](https://arxiv.org/abs/2503.06736)、[StanfordASL/oscbf](https://github.com/StanfordASL/oscbf)、[pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)、[agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros)。各上游许可证保留在子模块中。
