# 在另一台设备继续开发

仓库保留当前代码、配置、模型、全部现存实验数据和交接记录。支持目标是 Ubuntu 22.04 x86_64、Python 3.10、ROS 2 Humble；其他平台尚未验证。迁移本身不发送机械臂命令。

## 1. 获取代码和网格

```bash
git clone --recurse-submodules https://github.com/kaiwen123-yang/OSCBF-learn.git
cd OSCBF-learn
```

如果已经普通clone：

```bash
git submodule update --init --recursive
```

三个一级子模块和递归URDF子模块均固定到本机实际使用的版本，见 [sources.json](../requirements/sources.json)。使用子模块中保留的LICENSE；不要手动换成上游最新分支。

## 2. 建立环境

基础工具：

```bash
sudo apt-get update
sudo apt-get install git python3.10-venv build-essential
bash scripts/bootstrap_new_device.sh
```

脚本建立 `envs/piper-sdk`、`envs/oscbf`，按导出的实际版本安装，再运行作者Panda速度QP离线检查。它不启用CAN，不导入运动执行入口。
可通过 `PIPER_PYTHON=/path/to/python3.10` 指定创建环境的Python。

已装ROS 2 Humble且需要重新规划时：

```bash
bash scripts/bootstrap_new_device.sh --with-ros
```

ROS安装按 [官方Ubuntu说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 完成。项目脚本随后安装MoveIt等列明的apt依赖、创建基于系统Python的 `envs/ros-humble`、构建工作区。ROS环境继续使用Ubuntu的numpy/scipy，算法环境单独使用numpy 1.26.4 / JAX 0.4.30，不混用两个环境。

本机旧conda显式导出和pip/apt记录仍在logs中，供核对。新机采用Python venv重建同一Python大版本和已固定的pip依赖；不是复制conda目录，也不宣称系统包逐字节一致。

## 3. 无硬件验证

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/run_pick_place.py results/pick_place_plan_20261104_224921_126900/task.json
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/analyze_piper_diagnostics.py
```

第一条必须显示 `PLAN_PREVIEW_ONLY`。第二条复算11月6日四包数据并生成图，不连接CAN。

Piper模型仿真：

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_sim.py
```

模型回放（需要桌面显示）：

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/view_piper_plane_sim.py --result results/plane_sim_20261104_214857_157778
```

MoveIt只在重新规划时启动：

```bash
bash scripts/run_task_planner.sh
```

另一个终端：

```bash
bash scripts/ros_exec.sh python scripts/plan_pick_place.py
```

这是ROS域78的mock规划服务。原22:49成功任务计划已经归档，不必为了换机重新规划或改变路径。

## 4. 接回同一套实机

读 [PROJECT_STATE.md](../PROJECT_STATE.md) 和 [PICK_PLACE.md](PICK_PLACE.md) 后再操作。
已有配置对应20 mm方块、桌面低于底座安装面12 mm、抓取点前300/右80 mm、放置点前300/左80 mm，坐标是现场近似确认，TCP仍为厂家名义值。

USB-CAN串号绑定为 `004F00374548570120353731`，接口can0，1 Mbps。换电脑使用同一适配器应保持串号；若更换适配器，先确认设备身份和接口映射，再更新对应配置/脚本，不能跳过匹配。

通信启用由现场执行：

```bash
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up
./envs/piper-sdk/bin/python -I scripts/read_joint_state.py
```

第二条只读。根据实际姿态/模式确定后续动作，不根据旧日志自动回零。MOVE_J零位曾出现J2故障，reset会失电，电子急停可能下沉。当前换机工作不处理或重新验证这些物理动作。

## 保存范围与路径

- 全部现存 `results/`、相关 `logs/`、根目录ROS `log/` 随Git保存；包含失败、只读验证和软件准备记录。
- 上游SDK、算法、ROS及网格通过固定提交的递归子模块获取；不是把数百MB上游历史重复上传到本仓库。
- 不提交 `envs/`、ROS `build/install/log`、Python缓存、本地编辑器配置、运行锁、抓手册时得到的登录页面HTML。原文件留在本机；环境/构建过程关键日志和手册元数据保留。
- `models/` 运行路径已可重定位；STL引用相对URDF目录解析，算法加载使用当前checkout。历史结果中的绝对路径保留作为采集证据，不需要新机拥有 `/home/kaiwen/research/OSCBF-research`。
- `results/*/sources/` 是当次执行代码快照，供追查；实际执行入口仍是仓库顶层 `scripts/`。
- 仓库没有已确认的手机连续录像文件；不能从本机缺少的视频生成演示证据。

## 接手顺序

先核对11月6日分析中的CPV段末CAN突发及未复现的J2故障，再推进完整任务的实体障碍/OSCBF接入。阶段提交和关键证据见 [STAGES.md](STAGES.md)，最新判断始终写回PROJECT_STATE.md。
