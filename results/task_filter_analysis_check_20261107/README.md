> 2026-11-08：对应测试入口已按用户要求从当前代码删除；本目录保留的是此前记录，不是运行准入要求。

# 完整任务过滤器分析脚本的数值检查

分析入口是 `scripts/analyze_task_filters.py`，只读取已有结果并向指定独立目录写入 Markdown、JSON 和 CSV，不连接硬件、不运行控制器、不生成图片。

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I scripts/analyze_task_filters.py \
  results/<第一组> --compare results/<第二组> --out results/<独立分析目录>
```

只分析一个运行时省略 `--compare`。`--out` 不能位于任何输入运行目录内部；非空输出目录必须是该脚本之前为同一组输入建立的分析目录。输入原始文件不会改写。

## 输出

- `analysis.json`：任务完成、现场确认、介入比例/时间、速度修改量、介入窗口误差、QP/反馈/控制周期/CAN、逐约束负值和配对条件。
- `ANALYSIS.md`：文本报告，配对误差只使用共同相位域，不直接比较未对齐时间窗口的 RMS。
- `run_a_negative_values.csv` / `run_b_negative_values.csv`：每个严格负值的样本、阶段、相位、标签、单位、原值和样本保持时间。小负值也保留。若 NPZ 缺失，只保存 CSV 已记录的标量最小值，并明确不能恢复逐约束信息。
- `analysis_manifest.json`：输入目录与输出所有权记录。

`filter_constraints.npz` 的行数必须与 control.csv 对齐，列数必须与标签对应；还会核对 NPZ 汇总值与 CSV 的 min_h_m、min_physical_distance_m、min_applied_margin。对齐错误会标记为无效输入，而非把值当成可靠约束结果。

当前过滤器的 h/physical_h 包含几何距离 [m]、关节限位状态余量 [rad]、速度界 [rad/s]。分析按 `geometry_constraint_count` 和标签保留逐列单位，不把所有列的最小值混称为米。几何余量导数的 nominal/qp/applied margin 使用 m/s，关节与速度限制余量使用 rad/s。

时间统计按相邻同阶段控制行的前值保持估计；阶段间的夹爪/保持空白和末样本之后的未记录时间不外推。负值持续时间是这个明确采样规则下的估计，不是连续物理接触时长证明。所有 TCP/几何指标按模型/FK 解释，程序完成不替代现场确认。

## 配对条件

方法之外需要以下方法无关指纹：`path`、`geometry`、`parameters`、`controller`、`constraints`、`model`，放在 `filter_metadata.json` 的 `fingerprints` 字典。脚本同时重算保存路径、执行配置/倍率及记录的共同过滤器参数。只移除明确的方法选择字段；不同几何、参数、限速或模型不会被默认为相同。缺少指纹时标为未核实。

`strict_same_schedule` 使用实际 control.csv 中的阶段 elapsed/phase、阶段相对任务起点的时间偏移，以及 reference_schedules.json 的参考签名。两个声明文件相同不覆盖实际执行时间轴不同的事实。默认相位和时间容差均为 1e-6 s，报告保留实际差值；未完成运行不会得到该标签。

自适应配对在每个共同阶段的相交参考相位域上插值误差。重复相位取最后一个状态用于相位映射，并另报暂停时间、暂停时的最大误差等；介入对比使用双方相同的并集/交集相位窗口。没有直接比较两条未对齐时间序列的 RMS。一组配对不产生统计显著性或一般性优势结论；没有几何约束介入且速度未被修改时，只记录为任务接入证据。

## 小型数值检查

```bash
env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I \
  results/task_filter_analysis_check_20261107/check_analysis.py
```

`fixtures/` 只有少量人工构造数字行，均明确标记 `synthetic_numeric_fixture_not_hardware`；它们不是机器人仿真或实机结果。`fixtures/` 与 `reports/` 可由检查脚本完整重建，作为本地检查缓存忽略，不把这些人工数据混入实验提交；检查脚本和汇总结果保留。

已核对：相同实际日程判真、不同日程判假、复制相同声明日程不能掩盖不同 control.csv、几何指纹变化判为条件不一致、同一个误差-相位函数在不同时序下得到相同相位对齐指标、负 h 和微小负量化后余量保留、缺失 CAN 不写成零错误、程序完成不写成现场确认，以及禁止写入输入目录。检查结果在 `checks.json`，对应派生报告在 `reports/`。
