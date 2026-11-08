# 正常 CPV 命令组节拍验证

当前阶段：软件修复、离线回归及一遍完整实机任务已完成。`90a9482` 冻结版本在 19:07 的复跑中 CAN 错误为 0，用户确认物体转移成功。approach 阶段最大跟踪误差升至 0.992°，保留为尚未解释的局部变化，不能称所有指标均改善。

## 范围

`scripts/run_pick_place.py` 为完整任务创建 `PiperDevice(cpv_period_s=0.01)`。`piper_oscbf/pacing.py` 按单调时钟和上一组六轴命令的实际起始时刻安排下一组；一个组内仍依次发送六个关节，不在关节间等待 10 ms。正常运动、正常段末零速保持及夹爪阶段的正常 CPV 请求共用同一个调度状态。

迟到后以实际发送时刻继续安排下一组，不连续补发以追赶旧时间表。控制器在读取反馈前等待发送时隙，发送入口再次检查同一时隙。正常段末的第一组零速沿用上一组的时间，不重置调度器。

紧急停止、软件异常停止的第一组零速、使能反馈待确认时的第一组零速使用 `immediate=True`，不等待正常节拍。停止动作、原有 200 ms 反馈条件、原路径、名义跟踪律、限速及夹爪行为保留。

10 ms 是本项目名义 100 Hz 控制的设计值，不是已查明的厂家硬性要求。日志中的起止时刻是 SDK 发送调用组的单调时钟记录，不等同于物理 CAN 线上测得的逐帧时刻。

## 历史离线记录

2026-11-07 曾用假时钟记录发送函数修订前后的间隔。对应测试入口已按用户 2026-11-08 的要求删除；下面只说明已存历史记录，不是当前运行的前置门槛。

- 修复前：正常命令紧接零速的起始间隔为 0.120 ms，断言失败。
- 修复后：跨正常运动、零速和下一阶段的起始间隔均为 10 ms；超期不补发；紧急停止等待为 0；六轴分组与 0.001 rad/s 量化不变。
- 记录：`results/cpv_pacing_offline_20261107/`。

## 现场复跑

`90a9482` 的独立工作副本 `/home/kaiwen/OSCBF-cpv-validation` 只作为历史版本参照。当前执行入口为 `/home/kaiwen/OSCBF-learn`，包含后续时钟修订与用户要求的软件门槛清理。

准备条件：原桌面与底座布置，方块摆回右侧抓取点，运动范围清空，机械臂实际处于原计划零位附近。先读状态；若姿态或故障状态不符，不执行原计划，也不自动复位。

在 WSL 终端运行：

```bash
cd /home/kaiwen/OSCBF-learn
env -u LD_LIBRARY_PATH ./envs/piper-sdk/bin/python -I scripts/read_joint_state.py && \
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_pick_place.py \
  results/pick_place_plan_20261104_224921_126900/task.json \
  --speed-scale 1.5 --descent-speed-scale 1.5 --record-diagnostics --execute
```

动作包含张爪、接近、下降、抓取、抬升、运输、放置、撤离及 **return_initial**。两个速度倍率均为 1.5，关节速度上限 0.30 rad/s。运行时只有这个进程发送运动命令。正常结束保持 CPV 零速度及使能；Ctrl+C 使用原有零速加电子急停处理，可能下沉。没有新增自动回零、复位或异常后的重新使能。

当前日志写在主工作副本的 `results/pick_place_hardware_时间/`。`recording.json` 保存 Git 提交、代码修改状态和源码哈希；`commands.csv` 增加调度目标、实际起止、相邻开始/结束间隔、是否绕过节拍和发送完成关节数；`control.csv` 增加反馈年龄及实际发送组起止。

完成后读取原始记录：

```bash
./envs/oscbf/bin/python -I scripts/analyze_cpv_timing.py results/本次结果目录 \
  --baseline results/pick_place_hardware_20261107_180811_947624
```

检查实际物体转移、正常段末的发送组间隔、CAN 错误、控制周期、反馈年龄、跟踪误差、驱动故障及反馈等待。软件回归通过不替代实机通过。基线是同一台 WSL 的 18:08 记录；原生 Ubuntu 历史记录只作为额外参考。
