# Piper H 虚拟平面速度控制：第一组仿真

此入口不连接 CAN、不导入 SDK。状态来自理想速度模型 q[k+1]=q[k]+dt*u[k]，
所有末端轨迹来自模型 FK；不是实机轨迹，也不是动力学仿真。
回放使用官方带夹爪几何，夹爪固定在名义零开口。当前平面只约束法兰任务点，
没有添加全臂/夹爪碰撞约束；不能将此结果表述为整机避障完成。

## 入口

在项目根目录执行：

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_sim.py
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/view_piper_plane_sim.py
```

每次产生独立 `results/plane_sim_日期时间/`，保存 config.json、model.json、summary.json、
cbf.csv、oscbf.csv、comparison.png/pdf。失败或中断也保留数据。
`results/plane_sim_latest.txt` 指向最近一次运行，含失败运行；以 summary.json 的 status 为准。
动画按 C/O 切换两组记录，橙色为 CBF 路径、蓝色为 OSCBF 路径、灰色为参考、绿色球为参考位置。
Ctrl+C 或关闭窗口退出。动画直接回放关节状态，不启动 PyBullet 动力学求解。

## 控制与公平性

依据 [论文 v3](https://arxiv.org/pdf/2503.06736v3) IV-B 式 (18)–(28)，以及
本地作者提交 6082329f8e1d61ee8e875a63a5cd80a14d513e12 的
`oscbf/core/controllers.py::PoseTaskVelocityController`、
`oscbf/core/oscbf_configs.py::OSCBFVelocityConfig`。
模型与名义控制器直接调用作者实现；平面/关节约束矩阵按相同 CBF 公式显式构造，
使用已安装 qpax 求解，保留求解收敛标志。

两组唯一的控制律差异为二次目标 H：普通 CBF 使用 I；OSCBF 使用 JᵀWᵀWJ。
在线性项 -H*u_nom 下，目标为 0.5*(u-u_nom)ᵀH*(u-u_nom)，忽略常数项。
六轴满秩时 N=0，这是论文式 (25) 的化简，没有增加虚构的零空间。
本批不加正则项；记录 Jacobian 最小奇异值。零关节位姿不作为实验初态。
初态 `[0,30,-90,0,45,0]` 度仅用于本轮仿真，未核准从现场姿态直接运动到此处。

平面 h=b-nᵀp(q) >= 0，QP 中 nᵀJ_linear*u <= alpha*h。
同时加入官方关节范围的 CBF 和 ±0.15 rad/s 速度约束，两组均使用硬约束。
该速度是仿真参数，与 MOVE_J 的 10% 不是同一个量。
姿态目标为初始完整姿态，位置/姿态均作为跟踪目标，允许受约束后出现误差。
权重、增益、参考、初态、约束和速度上限均保存在同一 config/plane_sim.json。

虚拟平面距初态法兰 20 mm；参考法向平滑推进 40 mm，故意越过虚拟边界，
用于持续激活 CBF。沿平面切向以 2 mm/s 推进。两组使用完全相同的参考相位：
下一周期推进比例为两组实际模型切向速度/2 mm/s 的较小值，限制在 [0,1]。
跟不上时共同减速或暂停；故意受阻的法向分量不计入切向进度。
记录 progress_rate、phase_s；20 秒仿真结束可能尚未走完 20 秒参考，不能称任务完成。
这个共同相位规则专用于配对平面比较；完整抓放仍需持续受阻后的重规划/失败处理。

统计窗口是两组名义速度同时违反平面 CBF 的共同采样索引，避免各选不同时间段。
记录完整 CSV，包括无约束阶段；不预设 OSCBF 优势。
切向误差为平面投影后的二维位置误差范数；姿态误差为旋转矩阵测地角。
dt=0.01 s，逐周期显式更新关节状态；连续 CBF 在采样保持下可能有数值越界，
不裁剪关节或末端位置来掩盖越界，报告实际最小 h 与 QP 约束残差。

JAX 两组分别预热后计时。compute_ms 含 FK、名义控制、QP、诊断与同步，
不含文件写入和设备收发，也不代表 100 Hz 实机控制周期已验证。
QP 容差为 1e-6：本机 qpax 0.1.4 对内点变量采用 sqrt(eps) 数值下限，
1e-8 容差在首个无约束状态已无法满足其 KKT 收敛判据，初次失败记录已保留。
这是数值求解容差，不是实机动作偏差阈值；没有修改依赖包源码。

## TCP 与实机待接部分

base_link 与 flange_link 采用同源官方 URDF。21:31 的零位关节反馈 FK 与控制器
法兰输出差约 0.00102 mm / 0.00100 度，只说明两套软件模型输出一致，不能视为物理精度测量。
原始 motor_position 与关节角不能混用。进入 CPV 时的反馈差异仍待核对。

用户确认官方夹爪。官方 MoveIt xacro 的 tcp_offset_xyz/rpy 默认仍是全零，
不能仅凭“官方夹爪”断言已经配置指尖抓取中心。
用户手册网页本次只取得外壳/标题，未读到正文 TCP 定义，不凭空填数值。
下一步完成 J1–J5 CPV 速度方向及反馈映射，明确停止行为，再把共享控制器
接入每周期实测状态；实机全臂边界、TCP、场景和抓取物体几何继续按原项目要求接入。
