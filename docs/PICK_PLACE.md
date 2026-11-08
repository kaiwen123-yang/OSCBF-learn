# 固定方块抓放入口

## 2026-11-08 当前完整 OSCBF 入口

从已确认的六轴零位开始，先复用原基线计划的 open/pregrasp（CPV 上限 0.20 rad/s），再运行非奇异工作区计划的完整 OSCBF 抓放（上限 0.10 rad/s），结束返回右侧预抓取姿态。方块仍按已确认场景在右侧，额外障碍为空。该入口不会回零、reset 或更改编码器零点。

```bash
bash scripts/run_oscbf_pick_place.sh           # 两个计划的纯文本预览，不连接 CAN
bash scripts/run_oscbf_pick_place.sh --execute # 现场授权后执行
```

准备和完整任务处在同一个持续 WSL 调用中，全程持有同一发送者锁；准备失败就结束，沿用原执行器的停止处理。此改动处理分开临时 WSL 调用间发生空闲退出的问题，不改变路径、控制律、阈值、夹爪、SDK 或硬件错误判定，也不代表大弯曲姿态 MOVE_J 故障已修复。实际完整任务结果以 PROJECT_STATE.md 顶部为准。

下文按日期保留历史基线说明；末尾“尚未接入”的文字描述当时版本。

## 2026-11-06：原零位全阶段1.5倍复跑及诊断

本轮已完成抓放复跑和MOVE_J零位保持采集。旧故障未复现；抓放包捕获段末零速CAN错误。
统一分析及四包索引见 `results/pick_place_hardware_20261106_220347_670955/ANALYSIS.md`，当前状态见PROJECT_STATE.md。

用户现场确认场景未变，当前只读状态为六轴0°、正常且使能。复跑原23:14失败速度条件：

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py results/pick_place_plan_20261104_224921_126900/task.json --speed-scale 1.5 --descent-speed-scale 1.5 --record-diagnostics --execute
```

两个倍率均为1.5；只写`--speed-scale 1.5`会使approach/lower仍为1倍，与原失败条件不同。
仍由用户现场启动，唯一运动发送者为run_pick_place.py。正常结束保持CPV零速；Ctrl+C请求零速及电子急停，可能下沉。

新增`--record-diagnostics`在原结果目录保存：
- `raw_can.csv`：独立SocketCAN接收的CAN原始帧，含接收/错误标志、CAN时间及单调接收时间。SDK关闭本机回环，该文件不含SDK本机原始发送。
- `sdk_tx.csv`：从22:09第二次抓放开始记录SDK发送函数的ID、载荷、起止时间和异常；这是发送调用记录，不代表固件逐条执行确认。22:03第一包尚无此文件，原commands.csv速度发送记录保留。
- `states.jsonl`：约20 Hz的SDK缓存快照，含关节/控制器模式、六轴电流、驱动使能/故障、电压和温度，各消息时间戳分别保留。
- `events.jsonl`：阶段开始/结束、异常处理前、停止后的状态与调用栈。
- `console.log`、`recording.json`、`sources/`：终端、记录器统计/版本/命令/接口计数、当次控制代码。
- 原有`feedback.csv`、`control.csv`、`commands.csv`、`driver_feedback.csv`、`gripper.csv`、`task.json`、`summary.json`继续保存。

停止后额外只读3秒，不自动复位或重新使能。故障后先保留现场和日志，不连跑恢复命令。
控制律/计划未更改；11月4日晚已做过的瞬时使能反馈等待及软件错误零速停止仍保留，分析时需区分这些差异。

MOVE_J诊断入口为 `./envs/piper-sdk/bin/python -I scripts/home_zero_simple.py --target-deg 0 0 0 0 0 0 --record-diagnostics --execute`，
本次从零位附近开始，仅验证了切换/保持。它不是自动恢复入口；不要从未知姿态盲跑。

以下保留11月4日历史说明，当前进度以PROJECT_STATE.md顶部为准。

**最终交接：23:52原零位慢速计划完整复跑成功，用户要求结束本轮。**
证据 `results/pick_place_hardware_20261104_235251_893243/`，77.086秒，11阶段完成，
实际抓放由用户确认，驱动全程0x40，无异常等待/急停。方法为MoveIt路径＋实测关节CPV速度跟踪，
上限0.20 rad/s、前馈0.10 rad/s，未用OSCBF。完整逐阶段参数与原始日志说明见该目录summary.json。

复跑所用完整任务命令（须先确认已经在相应起点，不能从急停/未知姿态盲跑）：

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py results/pick_place_plan_20261104_224921_126900/task.json --execute
```

最后一次回零采用已规划CPV过渡 `results/pick_place_plan_20261104_234828_184708/task.json`，
从[0,20,-50,0,30,0]°准备姿态到零位附近，再保持CPV运行上述任务。**不要把它替换成
此前发生J2故障的MOVE_J零位保持。** 视频是否保存尚未单独确认；失败成因留待明天，详见PROJECT_STATE.md。

以下是历史过程记录，以顶部最终交接为准。

**23:36最新决定：今天不再排查零位，使用准备姿态始末计划录制完整实机视频。**
准备姿态[0,20,-50,0,30,0]°已有实机到位证据；完整任务计划为
`results/pick_place_plan_20261104_232956_473856/task.json`，执行时用 `--speed-scale 1.5`，
approach/lower默认1倍。当前正等待清故障、单独复位和准备姿态到位，不能从急停状态直接执行。
空爪回零仍出现J2 0xA4，具体成因留待明天；不要再用旧零位入口反复试。

**最新：23:15全阶段1.5倍尝试在lower触发使能反馈检查后急停，当前待分步恢复。**
不要直接重放下面任务命令。故障瞬间驱动状态未保存，根因未确认；当前已增加原始驱动
状态记录且没有放宽检查。后续 `--speed-scale 1.5` 默认只加快空中运动，approach/lower
使用 `--descent-speed-scale 1`（默认）保留之前成功的下降速度。具体恢复进度见PROJECT_STATE.md。

**23:10：首次完整实机抓放已成功，用户确认“很完美”。**
证据 `results/pick_place_hardware_20261104_230919_470403/`；11阶段结束并返回近零位，
保持CPV使能，方块已放到左侧。恢复和取出方块步骤已完成，不必重复。
用户要求加速：将方块摆回右侧后，用同一计划增加 `--speed-scale 1.5 --execute`
完整复跑（关节速度上限0.30 rad/s，保存原计划几何和新的执行参数）。加速现场结果待确认。

以下为此前恢复说明：
恢复已拆分：`recover_task_hold.py --reset-only --execute` 只reset，不使能或发送目标。
用户已完成本次reset及 `home_zero_simple.py --execute` 回零，不必重复。
可对22:49计划执行 `--stop-after open --execute` 单独张爪，取出并摆回方块；
随后不带 `--start-at` 从零位执行完整任务。旧的approach续跑起点已失效。
当前实际状态以 PROJECT_STATE.md 最新交接为准。

22:49更新：预抓取已执行，用户确认最终位于方块上方、指尖距桌面约120 mm。
该次末尾CPV→MOVE_J切换出现明显额外移动，已删除此切换；各段末尾及夹爪操作期间
维持零CPV速度，正常结束保持CPV模式，尚不能把零速称为独立位置保持保证。
新计划 `results/pick_place_plan_20261104_224921_126900/task.json` 包含align阶段，
从当前预抓取位置下降到桌面上方40 mm再停，现场对齐后才继续approach。
用户已要求加速：新计划关节速度上限0.20 rad/s、路径前馈上限0.10 rad/s。

此前七段 MoveIt 离线规划已通过；最新含align的八段路径已完成一次完整实机抓放。
`results/pick_place_plan_20261104_223514_486775/` 是未测量候选场景，不能直接执行。
已知物体为20 mm塑料方块；用户已确认安装面比桌面高12 mm。
22:44现场确认后的计划是 `results/pick_place_plan_20261104_224413_864865/task.json`，
XY接近建议值但存在未量化偏差；先用 --stop-after pregrasp 现场对齐，不直接下降。

## 现场布置与规划

以底座安装面中心为原点，零位机械臂指向为+X，向上+Z。
候选抓取点 X=300 mm、Y=-80 mm；放置点 X=300 mm、Y=+80 mm。
站在底座后方沿+X看时，-Y为右、+Y为左。请实测坐标和桌面高度后更新
`config/pick_place.json`。当前 scene_status=operator_confirmed_approximate；精确测量后才使用 measured。
TCP采用厂家标准夹爪尖端名义 `[0,0,0.1425] m`，不是现场标定结果。
尖端抓取高度为桌面上5 mm，接近/抬升高度为桌面上125 mm。

规划不连接CAN。终端1：

```bash
bash scripts/run_task_planner.sh
```

终端2：

```bash
bash scripts/ros_exec.sh python scripts/plan_pick_place.py
```

记录输出 RESULT_DIR。序列为张爪→接近→直线下降→抓取/附着→直线抬升→运输→直线下降
→释放/更新物体位置→直线撤离→返回初态。运输可通过 obstacles 列表添加实测箱体几何。
当前 obstacles 为空，不能称为绕障成果。服务在独立ROS域78、mock模型运行，不启动厂家实机节点。

## 实机入口（用户现场启动）

将下列 `<结果目录>` 替换为**实测场景重新规划后**的 RESULT_DIR。

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py <结果目录>/task.json
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py <结果目录>/task.json --stop-after pregrasp --execute
```

第一条仅预览；第二条张爪40 mm并低速沿规划到抓取点上方，保持位置。
查看夹爪朝向、尖端距桌面约125 mm、方块是否位于双指间正下方。
核对后继续下降—抓取—运输—放置—返回：

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py <结果目录>/task.json --start-at approach --execute
```

首次几何与抓取验证完成后，去掉 --start-at / --stop-after 可跑完整连续序列。
六轴速度上限以所用计划配置为准；最新为0.20 rad/s，前馈保留一半余量用于实际关节跟踪修正；夹爪力1 N。
使用全部规划路径点，不能跳过中间点仅发送终点。当前位置必须对应所选阶段的规划起点。
只有此脚本发送机械臂和夹爪命令；不要同时启动厂家ROS实机控制器或平面实验。
Ctrl+C请求零速和电子急停；必要时使用现场急停。正常结束发零CPV速度并保持使能，不切换MOVE_J。

记录在 `results/pick_place_hardware_时间/`：真实反馈、速度命令、参考/跟踪记录、夹爪反馈、
完整计划及summary。夹爪宽度只能提供接触线索，实际抓取/转移成功由现场和视频确认。
该执行器是 **MoveIt规划 + 普通关节速度跟踪**，尚未接入OSCBF或全臂CBF约束。
