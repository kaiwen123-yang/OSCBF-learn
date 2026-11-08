# 切回原生 Ubuntu 22.04：同计划完整 OSCBF 抓放

交接日期：2026-11-08，Asia/Shanghai。当前分支 `codex/wsl-setup`；固定接续标签 `native-oscbf-retest-20261108`。

本次只转移代码和实验记录，复用原生 Ubuntu 已运行正常的 SDK/算法环境。不要运行 `bootstrap_new_device.sh`、pip/conda 安装、ROS 重建、内核替换、固件升级或 WSL 连接脚本。执行保存计划不需要启动 MoveIt/ROS 实机节点。

## 1. 在原生系统取得固定版本，保留旧目录和环境

下面从 GitHub 建立独立源码目录，原 `/home/kaiwen/research/OSCBF-research` 的代码、日志和环境不被覆盖。目标目录应为这次新建的目录；若已有同名目录，保留它，换一个新的目录名。

```bash
git clone --recurse-submodules --branch native-oscbf-retest-20261108 \
  https://github.com/kaiwen123-yang/OSCBF-learn.git \
  "$HOME/OSCBF-native-20261108"
cd "$HOME/OSCBF-native-20261108"
git switch -c codex/native-ubuntu-retest-20261108
```

标签固定代码/计划/记录；新本地分支用于之后提交原生实测结果。不要对旧工作区执行 reset、clean 或强制覆盖。子模块仍取 `requirements/sources.json` 中固定版本。

## 2. 复用已有 Python 环境

下面的解释器路径来自 11 月 6 日原生 Ubuntu 实测记录，该环境已经同时包含 OSCBF 和 pyAgxArm SDK。切换后的实际目录若不同，只修改这个 shell 变量为原先可用环境的 Python 路径，不创建或重装环境。

```bash
export OSCBF_PYTHON=/home/kaiwen/research/OSCBF-research/envs/oscbf/bin/python

env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$OSCBF_PYTHON" -I -c 'import sys, numpy, jax, qpax, pyAgxArm; print(sys.executable); print("numpy", numpy.__version__, "jax", jax.__version__); print("pyAgxArm", pyAgxArm.__file__)'

bash scripts/run_oscbf_pick_place.sh
```

最后一条只打印两份计划，必须显示 `PLAN_PREVIEW_ONLY`，不会连接 CAN。预期原环境 Python 3.10、numpy 1.26.4、JAX 0.4.30，SDK profile v189。实际原生实测会把解释器、版本和源码快照写入新的 `recording.json`。

`OSCBF_PYTHON` 仅指定这次进程使用的已有解释器，未修改环境包、全局 PATH 或 shell 配置。若上述路径不存在或导入失败，先定位原先可用环境，不用安装命令掩盖环境差异。

## 3. 原生 USB-CAN 与只读状态

同一适配器：序列号 `004F00374548570120353731`，接口 `can0`，1 Mbps。原生系统直接接 USB，不使用 usbipd 或 `wsl_can_up.sh`。

```bash
ip -details link show can0
udevadm info --query=property --path=/sys/class/net/can0
```

核对串号。若 `can0` 已 UP 且 bitrate 1000000，保持现状；若接口 DOWN，在没有其他控制发送者时执行：

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

然后运行只读采集和一次固件查询：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$OSCBF_PYTHON" -I scripts/read_joint_state.py

env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$OSCBF_PYTHON" -I scripts/read_firmware.py
```

第一条没有机器人发送指令；第二条只查询固件，不使能、复位或运动。之前固件为 `H-V1.2-1 / S-V1.9-1`。若 `can0` 不存在或只读反馈异常，先解决连接，不从旧日志代入状态。

**当前不能假定已经在零位：** 最后一遍 WSL 在左侧撤离时中断，现场报告失能，之后没有自动回零。必须以这里新产生的关节反馈选择下一入口。如果不是下面两个已定义起点，只读到此为止，基于新的姿态单独处理恢复；不要把 `home_zero_simple.py --execute` 当作任意姿态自动恢复命令，也不要自动 reset/使能。

## 4A. 已在六轴零位：准备后完整 OSCBF

先把方块从左侧放回原右侧抓取点，移开额外圆柱，双手/支撑物离开运动范围。底座、桌面和方块尺寸沿用已确认场景。

```bash
bash scripts/run_oscbf_pick_place.sh --execute
```

这条命令依次执行：

1. 原计划 `results/pick_place_plan_20261104_224921_126900/task.json` 的 open/pregrasp。普通 CPV 跟踪，上限 0.20 rad/s，两个倍率均为 1。
2. 原计划 `results/pick_place_plan_20261107_193629_186524/task.json` 的完整 OSCBF。上限 0.10 rad/s，两个倍率均为 1。

全程同一个发送者锁；准备失败就结束，不自动恢复或继续下一段。零位准备是 baseline，不被计作满秩 OSCBF 算法段。

## 4B. 已在右侧预抓取姿态：直接完整 OSCBF

计划起点约为 `[-14.93,94.11,-72.94,0,73.83,-14.93]°`，以保存计划的精确弧度数值为准。只有新反馈和现场状态对应这个起点时，执行这一条，**不再运行 4A**：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  flock -n /tmp/oscbf-piper-004F00374548570120353731.lock \
  "$OSCBF_PYTHON" -I scripts/run_pick_place.py \
  results/pick_place_plan_20261107_193629_186524/task.json \
  --method oscbf --speed-scale 1 --descent-speed-scale 1 \
  --record-diagnostics --execute
```

完整任务为 open → pregrasp → align → approach → grasp → lift → transfer → lower → release → retreat → return_initial。结束回到右侧预抓取姿态，**不是六轴零位**；没有附加自动回零动作。

## 5. 相同条件与停止方式

- 保存计划和几何不改：20 mm 方块；抓取 `[0.300,-0.080] m`，放置 `[0.300,+0.080] m`；桌面 z=−0.012 m；TCP 名义偏置 `[0,0,0.1425] m`；额外障碍为空。
- 抓放阶段统一上限 0.10 rad/s；准备上限 0.20 rad/s；不是历史全阶段 1.5 倍运行。
- 名义控制 kp=2、OSCBF 度量、CBF 增益、量化 0.001 rad/s、10 ms 发送节拍、原到位/反馈要求及 v3 末段推进修正全部保留。
- 夹爪开口指令 40 mm、闭爪指令 15 mm、指令力 1 N。夹爪/零速保持阶段沿用原流程，OSCBF 用于手臂路径运动段。
- 正常结束保持使能、零 CPV 请求，不切 MOVE_J。Ctrl+C 沿用零速度与电子急停请求；厂家电子急停为阻尼模式，可能下沉。通信断开时软件停止请求可能无法送达，需要现场处理。故障后不自动复位、重使能或回零。

参数与文件校验值仅作为对照记录，见 `config/native_ubuntu_retest_20261108.json`；未增加强制预检/哈希准入或其他软件门槛。

## 6. 新结果与既有证据

新结果自动写入当前 checkout 的 `results/pick_place_hardware_时间/`；零位准备和完整 OSCBF 各自保留一包。成功必须同时有全部 11 阶段、原始通信/控制记录和现场完整转移/返回确认。失败包同样保留，不重命名为成功，不覆盖 WSL 包。

本次重点对照：

- `results/pick_place_hardware_20261108_221712_950540/`：代码 f3df8a6，物体转移现场确认，9/11 阶段，撤离时反馈停止，随后 USB/IP 自行断开。
- `results/pick_place_hardware_20261108_220631_617216/`：末段推进修正前的 OSCBF 放置停滞，CAN 错误 0；含真实日志重放与修正解释。
- `results/home_zero_diagnostics_20261108_214707_082496/`、`...190550_767544/`：大弯曲姿态 MOVE_J 失败，CAN ACK 错误；与 USB/IP 故障分开。
- `results/home_zero_diagnostics_20261108_215829_867857/`：近零位回零成功，CAN 错误 0。
- `results/pick_place_hardware_20261107_190744_717616/`：CPV 连续节拍修复后的完整基线，物体转移现场确认、CAN 错误 0。

未归档连续视频，没有完整任务普通 CBF 配对。本次切换目的为同代码/计划/参数下的原生 USB-CAN 实测，不提前将差异全部归因于 WSL。历史记录中的绝对路径保留原样。
