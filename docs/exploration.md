# 参考初态探索、记忆发布与冻结评测

## 流程与结果来源

```text
人工恢复参考布置 → HOME → 观察并执行工具 → 保存最终观察
        ↑                                    ↓
        └── 失败经验保留在本会话 inbox ← 人工验收
                                             ↓ 成功
                         文件工具整理策略 → 发布 audit / recipe
                                             ↓
                                冻结快照 → 新初态评测 / zero-shot 对照
```

`review.json` 是真实成功判断的来源，保存操作者、时间、三项任务条件、人工干预次数和最终图像/状态校验值。`result.json` 中的 `physical_success` 根据该验收记录更新。模型 `finish(success)` 单独保存为规划器结论。

真机 `reference-0` 是人工定义的物体、盒子及相机布置。每次探索前需人工复位桌面，HOME 恢复机械臂关节位置。第一轮及后续轮次的初态图像均保存在各 attempt 的 `initial/`。

## 开始探索

先检查配置和依赖，再启动探索。探索会产生 HOME 和工具动作。

```bash
bash run_harness.sh check
bash run_harness.sh explore --scene-id reference-0 \
  --reference-description '红色圆柱和蓝色方块位于盒外，盒子与相机保持参考布置'
```

省略 `--reference-description` 时会提示填写。每轮输入 `ready` 确认恢复初态，输入 `q` 退出。默认持续到首次成功或操作者退出，可通过 `--attempts 3` 设置尝试次数。每轮模型请求次数与规划时间沿用原参数，默认不限。

首次尝试使用当前探索记忆。后续尝试还能读取同一会话 inbox 中的失败结果、人工判断和工作笔记，从而调整策略。不同会话拥有独立 inbox。

每轮结束后填写 `success`、`failure`、`hardware-error` 或 `interrupted`，并记录原因及执行中的人工干预次数。初态复位单独计入会话记录。人工干预后的物理完成会保留为 `physical_success=true`，并标记 `autonomous_success=false`；这种轮次的经验继续保留在探索历史中。任务专属成功 recipe 和主成功率使用无人工干预的自主成功结果。选择成功时，需要逐项确认以下三点。

- 红色圆柱和蓝色方块均在盒内。
- 其他物体均在盒外。
- 夹爪已松开并撤离，目标物体仍留在盒内。

自主成功验收后，脚本使用同一规划器配置，以只包含文件工具的独立 API 会话整理策略。该阶段读取本轮记录，将策略写入 `notes/strategy.json`。API 用量和时间保存在 `distillation.json`，计入探索成本。整理器可读取执行记录和文本笔记，其视觉依据来自运行时保存的观察描述及人工验收。

发布器根据独立的 `tool-calls.jsonl` 生成每个步骤的真实工具名和参考参数。策略必须覆盖全部成功动作，逐步包含前置条件、完成判断、失败恢复和新初态参数重绑定方法。失败调用保留在 audit 中。

## 中断恢复与独立验收

Ctrl+C 后保存结果和能够获取的最终观察。恢复操作使用输出中打印的运行目录。

```bash
bash run_harness.sh review /path/to/attempt
bash run_harness.sh distill /path/to/attempt
bash run_harness.sh publish /path/to/attempt
```

独立 `review` 默认交互式验收。已有外部人工验收记录时，可显式提交结果。`--conditions-confirmed` 表示操作者确认了上述三项条件。

```bash
bash run_harness.sh review /path/to/attempt \
  --outcome success --conditions-confirmed --notes '已核对最终场景' --interventions 0
```

验收记录不可被另一份判断静默覆盖。重复同一验收命令可恢复因中断尚未同步的 `result.json`。发布操作可重试，并验证已有任务文件和证据是否一致。

若策略整理未通过完整性检查，错误会指出遗漏的动作或条件。可重新执行 `distill`，也可人工核对 `notes/strategy.json` 后重新发布。失败、dry-run 或缺少最终证据的轮次均保留记录。

## 记忆目录

默认状态根目录为当前 checkout 的 `.runtime`。AGX 现有部署通过 `harness.local.env` 将新记录放在 `/data/harness-vla`。

```text
<state-root>/
├── exploration/<session>/
│   ├── session.json
│   └── attempt-001/
│       ├── initial/                 global、wrist、state
│       ├── final/                   global、wrist、state
│       ├── configuration.json       模型、控制参数、场景和代码版本
│       ├── result.json / review.json
│       ├── tool-calls.jsonl          执行器记录的工具调用事实
│       ├── notes/strategy.json       带条件和重绑定方法的策略
│       └── <cell>.json / <cell>_recipe.jsonl
├── memory/ur5/
│   ├── task-specific/               成功 audit、recipe 和图像证据
│   ├── global/ / task-family/       经原生 MemoryManager 合并的经验
│   ├── results/                     发布回执与文件校验值
│   └── _internal/inbox/<cell>/      同会话草稿和尝试历史
├── snapshots/                       显式冻结的独立副本
└── evaluation/                      新初态的单轮执行记录
```

成功发布使用原生 `MemoryManager.merge_memory`。此前旧运行目录中的失败 audit 和工具序列继续作为历史记录保存；新协议从人工成功验收过的轮次创建任务记忆。

## 冻结与新初态评测

以下示例在可移植仓库根目录运行。现有 AGX 部署将 `.runtime` 替换为 `/data/harness-vla`。

```bash
bash run_harness.sh freeze \
  --memory-dir .runtime/memory/ur5 \
  --output-dir .runtime/snapshots/pickplace-v1

# 有记忆评测
bash run_harness.sh evaluate \
  --snapshot .runtime/snapshots/pickplace-v1 --scene-id layout-1

# 同一布置与预算的 zero-shot 对照，复位后执行
bash run_harness.sh evaluate --zero-shot \
  --snapshot .runtime/snapshots/pickplace-v1 --scene-id layout-1
```

冻结操作创建独立副本和内容清单。评测开始及结束时检查全部文件及快照 ID，文件工具对快照只读。zero-shot 组的文件工具使用独立空记忆，快照仅用于核对固定配置。

两组核对 checkpoint SHA-256、模型与 API 地址、推理强度、任务指令、控制参数、预算、工位配置、执行代码和关键规划依赖版本。执行前读取 WebSocket 服务实际加载的 checkpoint 标识。配置变化会明确报错，并保存诊断结果。

为每个预先选定的布置分别执行两组，使用相同的物体初态、试验次数和预算。可在不同布置间交替两组顺序。场景 ID 与每轮初态图像共同保留实际布置证据。新模型或新配置需要重新形成对应的记忆版本。

## 汇总

```bash
bash run_harness.sh summarize .runtime --output .runtime/summary.json
```

汇总按协议、模型、完整配置、快照和运行模式分组，列出人工验收结果、总耗时、规划耗时、VLA 调用次数、API token、整理成本、人工干预、硬件错误和待验收轮次。组内存在待验收结果时，整体成功率保留为 null，避免将待验收轮次从分母中隐去。

探索和正式评测分别统计。比较两组时使用相同 snapshot ID、配置 ID 和场景 ID，并保留硬件错误及人工干预记录。
