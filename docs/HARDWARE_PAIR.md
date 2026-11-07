# 放大范围后的实机 CBF / OSCBF 配对

2026-11-04 22:36 已完成固定共同时序实机配对，见
`results/plane_pair_20261104_223600_152131/summary.json` 和 `comparison.png`。
固定参考入口（后续公平比较使用）：

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_pair.py --fixed-reference --execute
```

该开关让两组跟随同一外部参考时序，不按各自误差改时序，仅用于空旷空间的虚拟边界实验。
任务执行中的进度调节与此受控比较分开。两轮仍有实测初态差，应读取 summary 的具体值。
本次共同激活窗口切向RMS OSCBF/CBF为0.181/6.204 mm，姿态RMS为0.066/2.700°；
最小h为-0.645/-0.605 mm，不满足严格全程不越界。单次固定顺序不能代表统计结论。
控制CSV现在直接记录六轴名义控制及实际SDK输入相对名义值的修正。

22:31自适应时序组因CBF额外减速不作为同时间参考的公平比较。该组JSON序列化故障已修复；
恢复旧CSV可用 `scripts/summarize_plane_pair.py <结果目录> --operator-confirmed-return`，
只有现场确实确认正常完成/返回时才使用此标志，不会连接硬件。

以下为原自适应记录/重放说明：

前次闭环 `results/plane_hardware_20261104_221628_926533/` 完成 1200 周期，
所有 QP 收敛，周期中位数约 10 ms；最小 h=+0.4085 mm。
但 925/1200 周期参考推进比例为零，12 秒只推进 2.721 秒参考，法兰实际 FK 位移约
[2.591,0,1.567] mm，只有 24 个采样的名义速度触发平面约束。
因此它证明接口与闭环接通，尚不是充分的边界跟踪实验。

## 本次动作与入口

```bash
cd ~/research/OSCBF-research
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_pair.py
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./envs/oscbf/bin/python -I scripts/run_piper_plane_pair.py --execute
```

第一条 Python 命令只读取、打印目标并预热。--execute 才发起运动：

1. 从本次实测角度以 MOVE_J 10% 到 q+[0,20,-50,0,30,0]°。
2. 在准备后的实际法兰位置建立统一参考。+X 15 mm 处为虚拟平面，参考 +X 30 mm，
   +Z 切向 8 mm/s，12 秒最多参考上移 96 mm，六轴速度 QP 上限 0.10 rad/s。
3. 先运行 OSCBF 并保存完整参考位置/速度/相位时序；以 MOVE_J 10% 返回同一实测起点。
4. CBF 重放该参考，使用同一平面、初始姿态目标、模型、增益和速度约束。
5. 零速后 MOVE_J 10% 返回本次初始位置，保持使能。正常约 45 秒；额外暂停可能延长。

用户现场确认准备动作和上移范围空旷，只有本进程发送运动指令。
Ctrl+C/异常请求六轴零速及电子急停，可能下沉，必要时使用物理急停。
不操纵夹爪，不 reset，不改参数 Flash；仍只有法兰任务点的虚拟边界，无全臂碰撞保证。

## 进度修正与公平性

此前把量化后的速度比例递归乘进参考速度，小指令量化为零会锁住进度。
新逻辑在名义输入未受 QP 约束时正常推进；受约束且沿切向落后时，依据实测关节 FK
计算的切向滞后及原有比例增益连续减慢/暂停。切向超前不暂停正在追赶的参考。
这不是增加位置偏差急停阈值。

OSCBF 的位置和速度参考都存入 reference.csv，并记录 SHA256。
第二轮以自己的控制周期对同一已保存时序作线性插值，不重新创建法兰原点。
两轮目标起点相同，但仍记录各轮实测初态差值，不能说真实初态绝对相等。
若 CBF 需要比保存的时序更多减速，则允许它减慢，记录额外暂停，并将
matched_reference_schedule 标为 false；此时只保存描述性曲线，不输出公平配对 RMSE。
重放最多运行原时长的两倍；未重放完整的试次明确标记 reference_complete=false。

matched_reference_schedule=true 时，在两轮名义速度同时触发平面约束的共同时间窗口
统计切向/姿态 RMSE。成功或失败、无约束阶段均保留。单次固定顺序不证明普遍优势，
后续重复试验需考察顺序影响。所有末端数据均为关节反馈经 FK 计算，非外部计量。

数值目标整体缩放为 100（两组相同，精确最优解不变）。开发检查发现原来的缩放1000
使普通 CBF 在第189个数值采样达到 qpax 内点变量下限影响的收敛精度瓶颈；
调整后再核对，不修改求解器依赖。参数和数值核对见 config/plane_pair_hardware.json、
logs/plane_pair_numeric_check.json；数值核对不是实机结果。

结果在 results/plane_pair_时间/，包含两轮控制 CSV、完整反馈/命令、reference.csv、
参数/模型/summary.json、comparison.png；失败保留已取得的数据。
PIPER_PLANE_PAIR_FINISHED 仅表示序列结束，仍需看 status、reference_complete、
matched_reference_schedule、初态差、最小 h 和共同激活窗口。
