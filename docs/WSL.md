# Windows 与 WSL 的 Piper 开发环境

配置日期：2026-11-07（Asia/Shanghai）。这是独立的 WSL 工作副本，原生 Ubuntu 双系统中的文件没有迁移或改写。

后续实机验证：2026-11-07 18:08 已按用户要求完成一遍原路径、全阶段 1.5 倍的完整抓放，用户确认物体转移成功。任务耗时 56.011 秒，仍记录到 35 条段末 CAN 错误帧；详见 [本次通信分析](../results/pick_place_hardware_20261107_180811_947624/WSL_COMMUNICATION.md)。下文“本轮验证”描述的是此前环境配置阶段，运动复跑作为独立结果保存。

## 位置与打开方式

- WSL 发行版：`Ubuntu-22.04`，用户 `kaiwen`。
- 项目：`/home/kaiwen/OSCBF-learn`。
- Windows 访问：`\\wsl.localhost\Ubuntu-22.04\home\kaiwen\OSCBF-learn`。
- 基础提交：`2ea3f1f48d10591b2bcb770af704011b381910b7`。
- 本地配置分支：`codex/wsl-setup`。本轮没有向 GitHub 推送。
- Windows 当前任务目录内，`打开_OSCBF_WSL.cmd` 打开 VS Code 的 WSL 项目；`连接_Piper_WSL.cmd` 连接指定 USB-CAN、读取状态，然后进入项目终端。

也可以在 PowerShell 进入项目终端：

```powershell
wsl -d Ubuntu-22.04 -- bash /home/kaiwen/OSCBF-learn/scripts/wsl_shell.sh
```

## Python 与 ROS

沿用仓库固定的依赖及四个上游提交，建立了三个项目内环境：

| 环境 | 用途 |
|---|---|
| `envs/piper-sdk` | Piper SDK、只读设备状态及现有实机入口 |
| `envs/oscbf` | JAX 0.4.30、OSCBF、PyBullet 无界面仿真、原有控制代码 |
| `envs/ros-humble` | 基于 Ubuntu 系统 Python 的 ROS 2 / MoveIt 环境 |

ROS 工作区已构建。调用 ROS 工具使用 `bash scripts/ros_exec.sh ...`，保持系统 ROS 的 numpy/scipy 与算法环境隔离。算法沿用 CPU JAX 后端；没有安装或替换显卡驱动。

不需要从双系统 Ubuntu 复制 conda 环境，也不需要调整 `.bashrc`。项目终端只为当前会话激活环境。

## USB-CAN 连接

仅匹配 `1d50:606f`、序列号 `004F00374548570120353731`，接口固定为 `can0`、1 Mbps。Windows 的 BUSID 可能随插口变化，连接脚本根据序列号查找，不硬编码 BUSID。

在 Windows PowerShell 执行：

```powershell
& '\\wsl.localhost\Ubuntu-22.04\home\kaiwen\OSCBF-learn\scripts\connect_wsl.ps1'
```

连接过程包含：USB 共享状态检查、附加到 WSL、等待设备枚举、核对设备身份、配置 CAN、运行已有的 `read_joint_state.py`。不会使能、复位、回零或发送运动目标。已有活动 CAN 接口若比特率错误或 bus-off，脚本报错，不自动重置。

USB-CAN 附加期间请保持 WSL 终端或 WSL 开发会话打开；Windows 快捷入口在连接后自动进入项目终端。关闭所有 WSL 会话或重新启动 Windows 后，重新运行连接入口。

如果 Windows 本身显示“未知 USB 设备（设备描述符请求失败）”，需要先重新插接电脑端 USB，使 Windows 恢复识别指定序列号。脚本不会把未知设备当成 Piper，也不会靠重启机械臂恢复 USB。

## 驱动配置

原 WSL 内核 `6.6.87.2-microsoft-standard-WSL2` 有 SocketCAN，但缺少 `gs_usb` 和 CAN 接收卸载支持。

使用 Microsoft 官方 `linux-msft-wsl-6.6.87.2` 源码，提交 `427645e3db3a8896714f22a3d3fe0c3f7b317ad4`，按当前内核配置启用 `CONFIG_CAN_GS_USB=m`、`CONFIG_CAN_RX_OFFLOAD=y`。编译完整内核仅用于生成符号版本，没有安装或切换内核。

仅添加：

- `/lib/modules/6.6.87.2-microsoft-standard-WSL2/updates/oscbf/can-dev.ko`
- `/lib/modules/6.6.87.2-microsoft-standard-WSL2/updates/oscbf/gs_usb.ko`
- `/etc/udev/rules.d/70-oscbf-piper-can.rules`

原系统模块保留在原目录。项目模块删除了与 Microsoft 原始 vmlinux 不一致的可选 BTF 调试段，保留模块版本检查。没有使用强制加载选项。

模块副本及校验值放在 `.wsl/driver/`。WSL 的模块目录使用运行时 overlay；`wsl_can_up.sh` 会在 overlay 丢失时重新安装项目模块，并严格检查内核版本。将来如果 WSL 内核更新，需要为新内核重编译；脚本拒绝把旧模块装到新内核。

编译源码、配置、日志保存在 `.wsl/`，该目录不纳入 Git。现有 `.wslconfig`、网络模式、代理、启动配置和 shell 配置保持原值。

## 本轮验证

- 三个软件环境和 ROS 工作区准备完成；SDK 与算法环境 `pip check` 通过。
- 作者 Panda 速度 QP 检查通过，记录 `logs/oscbf_smoke_20261107_174754.json`。
- Piper 无界面仿真 20 秒、每种方法 2000 步、QP 全部收敛，记录 `results/plane_sim_20261107_174941_651333/`。仅代表仿真检查。
- 原抓放计划显示 `PLAN_PREVIEW_ONLY`，没有执行动作。
- MoveIt mock 服务启动及 FK 查询通过，检查期间 CAN TX 增量为 0；记录 `.wsl/logs/moveit-check.json`。
- 首次真实反馈检查成功：`logs/joint_state_20261107_174649_039622.json`，998 组、199.67 Hz、最大间隔 5.499 ms，接收错误和丢包为 0，TX 增量为 0。
- 随后的 USB 重连出现无反馈，后续 Windows 枚举报告设备描述符请求失败。用户重新插接电脑端 USB 后恢复；`logs/joint_state_20261107_180030_848649.json` 验证 1000 组、199.93 Hz、最大间隔 5.52 ms，无丢包、接收错误或 TX 增量。失败日志全部保留，不宣称任意热重连均可靠。
- 双向通信验证：`logs/firmware_20261107_180127_240500.json`。固件查询收到 `H-V1.2-1 / S-V1.9-1`，SDK 解析为 `v189`；1 个查询帧成功发送、12182 帧接收、收发错误及丢包均为 0。该查询不改变机械臂模式或目标。
- 按用户要求停止图形窗口排查；不把 WSLg 窗口列为已通过项目。

没有执行使能、回零、复位或机械臂运动。100 Hz 实机运动闭环仍需结合实际运行日志验证，不能用上述 QP 耗时或接收频率替代。原项目 CPV 段末 CAN 错误和 J2 故障的研究结论保持原样。

## 改动与记录

既有 `scripts/`、`piper_oscbf/`、`config/`、`models/` 和历史结果已用配置前 SHA-256 清单核对。安装入口产生的新环境清单单独留在 `.wsl/logs/current-environment/`，归档清单和历史 `latest` 指针已恢复原值。

本轮新增 WSL 连接/终端入口、精确序列号 udev 规则、项目编辑器配置、验证记录和本说明。APT 仅补齐项目所需的 111 个软件包及依赖，没有升级或删除既有软件包。

参考：[Microsoft USB 连接说明](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)、[VS Code WSL 开发](https://code.visualstudio.com/docs/remote/wsl)。
