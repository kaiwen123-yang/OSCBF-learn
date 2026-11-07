# 第一段 Piper H 实测状态 OSCBF 闭环

实机动作由用户启动。当前是单组工程验证，尚不是普通 CBF/OSCBF 公平配对实验。
前置证据：`logs/cpv_axes_20261104_215945_973356.*`，J1–J5 SDK 正负速度与
关节反馈方向一致；J6 已在先前独立序列验证。J3 零速阶段曾余移约 0.089°，
不得把零速指令当成瞬时制动。此次全轴零速度进入 CPV 没有观察到关节角跳变；
这不证明所有姿态/固件均如此，也不能把原始 motor_position 直接代入 URDF。

## 运行

项目根目录，给已有 OSCBF Conda 环境补同版本 SDK/CAN 依赖：

```bash
./envs/oscbf/bin/python -I -m pip install --no-build-isolation ./src/pyAgxArm 'python-can==4.6.1' 'typing-extensions==4.16.0' 'wrapt==1.17.3'
```

先读取、预览并预热（不发控制命令），然后在现场启动动作：

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_hardware.py
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_hardware.py --execute
```

执行序列：读取 q_initial → JAX 预热 → 使能/以当前位置为初始目标 → MOVE_J 10%
到 q_initial+[0,10,-10,0,10,0]° → CPV 六轴零速 1 秒 → 以该时刻实测 q
建立法兰参考和虚拟平面 → 12 秒 OSCBF → 零速 1 秒 → MOVE_J 10% 返回 q_initial。
正常结束保持使能。执行期间只有这个进程发送实机运动命令，不同时启动 ROS 实机控制器。
准备目标及路径需由用户在现场确认；本入口不提供全臂碰撞检测，不操作夹爪。

Ctrl+C、求解失败、通信中断、失能或控制器故障：请求全零速度，再电子急停。
电子急停可能下沉；现场必要时使用物理急停。无自动 reset、失能或故障恢复。
没有添加人为角度偏差/速度误差判停；MOVE_J 到位以控制器自身反馈标志为准。
进程强杀或 CAN 故障可能使停止报文无法送达，软件停止不等于独立硬件看门狗。

## 控制实现

- 与仿真共用 `piper_oscbf/plane.py`，硬件适配 `piper_oscbf/device.py`。
- 参数 `config/plane_hardware.json`。base_link 下 +X 为平面法向，+Z 为切向。
  平面在初始法兰 +X 3 mm，参考向 +X 平滑推进 6 mm，并以 +Z 0.5 mm/s 向上。
  关节速度 QP 上限 0.02 rad/s，跟踪完整初始姿态。边界为虚拟任务点约束，非实体障碍。
- 每次计算使用新完整 0x2A5/6/7 关节反馈，未积分指令当状态。读取线程只收帧，
  主线程是唯一命令发送者。以 10 ms 为目标周期，记录实际周期与反馈年龄。
  反馈/计算输入过期 0.2 s 时停止；JAX 同步完成后才下发。预热在任何运动之前。
- 实测准备姿态用于定义参考原点/姿态/平面，作为已预热函数的动态参数，不重新编译。
  参考进度按当前实测状态处 QP 计算、经指令量化后的切向速度减慢/暂停；这是模型预测速度，
  不声称是独立测得的末端速度。限时结束后记录实际走完的参考相位；无进展不伪称完成路径。
- 固定版本 SDK 的 CPV `sp` 指令分辨率是 0.001 rad/s。
  发送前按同样规则取整并记录浮点 QP 输出与实际 SDK 输入。
  线性约束预留由半个量化单位推导出的误差：平面为 0.0005*||nᵀJ_linear||₁；
  关节 CBF/速度上下界为 0.0005 rad/s。这不是人为的运动误差阈值。
- 硬件配置中二次目标整体乘 1000，改善弱方向的数值求解精度；硬约束 QP 的精确最优解不变。
  仿真默认缩放为 1、无量化补偿，其已有配置行为保留。
- 实际驱动响应延迟、反馈量化、模型误差尚未纳入 CBF 严格保证。本轮观察真实 h，
  不裁剪反馈或隐藏负值；空旷空间中的虚拟边界可用于测量这些影响。

## 记录与后续

`results/plane_hardware_时间/` 保存 config/model/summary.json、feedback.csv、commands.csv、
control.csv、hardware.png。失败保留已收数据，summary.status 区分 preview/completed/failed/interrupted。
读取参数/依赖版本自动进入记录。末端位置和姿态误差均由实测关节角做 FK 得到，非视觉或外部计量。
`PIPER_PLANE_HARDWARE_FINISHED` 表示执行序列完成，不自动证明约束全程满足或跟踪精度合格。

先看这一次实测结果，再做配对比较。虽保留 `--method cbf` 开关，分别直接运行两个方法时
各自重新测量原点、各自调节参考进度，不能声称二者使用相同参考；公平对比需要固定同一
场景/参考原点/姿态，并安排共同时间参考或同参考相位采样，核对实际初态差异。
