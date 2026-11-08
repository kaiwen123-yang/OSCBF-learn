# 阶段、提交与证据

本次按六组内容提交到main，每组有 `stage-01-...` 到 `stage-06-...` 标签。它们是2026-11-07的迁移分组，不伪造11月4日/6日的Git开发历史。中间标签是内容快照，完整开发环境以main及递归子模块为准。

| 迁移组 / Git标签 | 内容 | 对应项目阶段 |
|---|---|---|
| 01 `stage-01-environment` | 固定上游源码、Python依赖、换机安装与目录说明 | 基础设施 |
| 02 `stage-02-hardware-basics` | 只读状态、CAN、MOVE_J/CPV基础运动与恢复代码及原始记录 | A |
| 03 `stage-03-model-simulation` | ROS模型导出、作者示例、Piper模型仿真及记录 | C |
| 04 `stage-04-oscbf-hardware` | OSCBF/CBF实测闭环、同参考平面对比与图 | D |
| 05 `stage-05-pick-place` | MoveIt规划、夹爪、连续抓放、成功与失败包 | B部分 |
| 06 `stage-06-diagnostics-handoff` | 全量CAN诊断、对比分析、最新状态与迁移验证 | F部分；E待完成 |

## A：基础实机运动，已完成

入口：`scripts/read_joint_state.py`、`cpv_j6_demo.py`、`cpv_axes_demo.py`、`home_zero_simple.py`。

- [真实反馈](../logs/joint_state_20261104_201906_815076.json)
- [基础回零](../logs/home_zero_simple_20261104_211038_869323.json)
- [J6 CPV](../logs/cpv_j6_20261104_211713_960091.json)、[多轴CPV](../logs/cpv_axes_20261104_215945_973356.json)

历史失败及恢复同样保留；A通过不代表任意姿态回零都可靠。

## B：MoveIt完整抓放基线，部分完成

入口：`scripts/plan_pick_place.py`、`run_pick_place.py`，配置 `config/pick_place.json`。

- [已执行的原零位计划](../results/pick_place_plan_20261104_224921_126900/task.json)
- [首次完整成功](../results/pick_place_hardware_20261104_230919_470403/summary.json)
- [慢速重复成功](../results/pick_place_hardware_20261104_235251_893243/summary.json)
- [1.5倍现场确认成功](../results/pick_place_hardware_20261106_220347_670955/summary.json)

已经实际抓取—抬升—运输—放置—返回。场景没有额外实体障碍，未完成已知障碍绕行；当前执行是MoveIt路径＋CPV关节跟踪。

## C：论文仿真与Piper模型迁移，已完成

- 官方Panda示例：[离线QP记录](../logs/oscbf_smoke_20261104_212035.json)，`scripts/run_oscbf_velocity.sh`。
- Piper模型：[模型元数据](../models/piper_h_oscbf_model.json)，[理想速度仿真](../results/plane_sim_20261104_214857_157778/)。

这两类结果是仿真，不替代真机反馈；Piper TCP未作外部标定，FK不是相机测量。

## D：OSCBF实机控制，已完成

入口：`scripts/run_piper_plane_hardware.py`、`run_piper_plane_pair.py --fixed-reference`。

- [首次反馈闭环](../results/plane_hardware_20261104_221628_926533/summary.json)：仅接口/约束介入证据，参考推进受限。
- [固定共同参考成对实验](../results/plane_pair_20261104_223600_152131/summary.json)
- [对比图](../results/plane_pair_20261104_223600_152131/comparison.png)

真实CAN关节输入计算QP并进入CPV执行接口。OSCBF/CBF激活窗口切向RMS 0.181/6.204 mm；最小h为-0.645/-0.605 mm，保留负值，不宣称严格不越界。

## E：完整任务接入OSCBF，尚未完成

2026-11-08 已接入同一完整执行器，并实际完成无新增障碍的 OSCBF 抓取、运输和放置，用户确认物体在左侧；但撤离阶段发生传输中断，返回未完成，完整序列仍待完成。证据见 [22:17 记录](../results/pick_place_hardware_20261108_221712_950540/ANALYSIS.md)。已知实体障碍任务及完整 CBF 配对尚未执行；D 的独立虚拟平面对比不能代替它们。

## F：结果与演示，部分完成

已有重复抓放记录、平面对比、参考/反馈/约束/修正/运行时间、完整计划和复跑入口；没有已归档连续视频或完整OSCBF任务对比。

- [2026-11-06统一诊断](../results/pick_place_hardware_20261106_220347_670955/ANALYSIS.md)
- [四包对比数据](../results/pick_place_hardware_20261106_220347_670955/diagnostic_comparison.json)
- [J2旧故障](../logs/driver_fault_20261104_232415.json)
- [旧1.5倍中止](../results/pick_place_hardware_20261104_231423_689682/summary.json)
- [旧空夹/异常J6帧](../results/pick_place_hardware_20261104_233916_927953/summary.json)

保留尚未确认的因果关系。下一首要工作及历史现场确认见 [PROJECT_STATE.md](../PROJECT_STATE.md)。
