# VeraPlan 与 π0.5 对接

VeraPlan 负责观察、动作条件检查、技能选择、结果判断和修复步骤。此仓库提供双相机与机器人状态读取、语言条件 π0.5 推理、有限时动作段，以及暂停和停止接口。控制执行继续使用同一个 RTDE、夹爪和相机所有者。

接口沿用现有图像输入与机器人基座位姿表示。手眼标定由后续实验安排。

## 入口与文件

| 文件 | 用途 |
|---|---|
| `planner_bridge.py` | `open_session`、`Pi05Session.observe/act/pause/stop/wait/status` |
| `examples/veraplan_minimal.py` | 真相机/RTDE 观察、π0.5 动作段、结束后重新观察 |
| `examples/infer_observation.py` | 将调用方提供的图像、状态和语言发给模型服务，保存 50×7 动作 |
| `policy_rollout.py` | 硬件连接、HOME、30 Hz 控制和 hold |
| `rollout_io.py` | 相机缓存、动作块、异步模型客户端和轨迹执行 |
| `serve.py` | 模型加载、编译、预热和本机 WebSocket 服务 |

## 环境和启动

实测设备为 Jetson AGX Orin 64GB、Ubuntu 22.04、JetPack 6.2 / L4T R36.4.3、CUDA 12.6。策略环境为 Python 3.10、Jetson Torch 2.11.0、Triton 3.6.0。UR5 为 CB3 / PolyScope 3.15.8，两路 RGB 相机为 1280×720、30 FPS，夹爪为大寰 PGC-300-60 / Modbus RTU。

按仓库 README 的安装步骤创建环境，将 `config/ur5.example.toml` 复制为 `config/ur5.local.toml`，填写机器人 IP、global/wrist 设备路径与夹爪串口。模型所有者单独提供完整 checkpoint，接入后启动本机服务。

```bash
git clone https://github.com/EverNightCN/ur5-pi05-inference.git
cd ur5-pi05-inference
bash setup.sh
# 编辑 config/ur5.local.toml，并将完整 checkpoint 放到本机
python3 tools/configure_checkpoint.py /path/to/checkpoint
bash run.sh tools/make_warmup_sample.py
python3 tools/install_service.py
bash start_inference.sh start
bash start_inference.sh status
```

安装依赖与 JetPack 匹配方法见 README。`install_service.py` 生成手动启动的 user service，首次启动完成模型加载、编译和三次预热后开放 `/healthz`。后续动作段复用已驻留的模型。服务释放命令为 `bash start_inference.sh stop`。

在已有 AGX 部署上，也可以连接其已启动的策略服务，使用 `--host`、`--port` 指定端点。当前条件式推纸盒模型的设备入口为 `run_push_pickplace.sh warmup`，端口为 `8002`，该工位脚本由设备部署目录提供。公开仓库的通用服务默认使用 `8000`。

每个真实工位由一个控制进程持有相机、RS485 和 RTDE 输入寄存器。从 PICO 采集切换到推理前，先结束已有 rollout，并释放采集进程的连接。

```bash
sudo systemctl stop ur5-teleop.service
```

接口依赖本机 π0.5 运行环境。规划器的模型与 API 配置由 VeraPlan 侧管理。

## 观察输入与机器人反馈

```python
with open_session(args) as session:
    current = session.observe()
    observation = current['observation']
    global_rgb = observation['observation/image']
    wrist_rgb = observation['observation/wrist_image']
    state = observation['observation/state']
    feedback = current['feedback']
    # 将 global_rgb、wrist_rgb 和 feedback 接入 VeraPlan 的视觉状态检查。
```

| 字段 | 类型与含义 |
|---|---|
| `observation/image` | global RGB，`uint8`、H×W×3 |
| `observation/wrist_image` | wrist RGB，`uint8`、H×W×3 |
| `observation/state` | `float32`、13 维，TCP 6 + 关节 6 + 归一化夹爪 1 |
| `prompt` | 本次模型语言条件，字符串 |
| `observed_at` | 本机 monotonic 秒，当前观察中最早的数据时间戳 |
| `feedback.timestamp` | 测量反馈的本机 monotonic 秒 |
| `feedback.tcp_pose_base` | 基座系 `[x,y,z,rx,ry,rz]`，米与旋转向量弧度 |
| `feedback.joints_rad` | J1–J6，弧度 |
| `feedback.gripper_position_raw` | 0–1000，0 闭合、1000 全开 |
| `feedback.gripper_state` | 0 运动、1 到位、2 抓到物体、3 掉落 |
| `feedback.robot_mode/safety_status/runtime_state` | 控制器返回的设备与程序状态 |

模型状态中的 TCP 旋转向量使用训练数据对应的正 Ry 等价分支。`feedback.tcp_pose_base` 保留控制器原始表示，二者可能存在等价旋转向量数值差异。状态读取使用 RTDE 后台接收缓存，双相机持续取流。观察和反馈分别带有时间戳，调用方可据此整理闭环记录。

## 将语言与当前画面传给 π0.5

`session.act(instruction, seconds)` 每次从当前双相机和机器人状态构造观察，将 `instruction` 写入 `prompt`。模型返回完整的 50×7 动作块，控制器按动作时间索引执行。

```python
result = session.act(instruction=training_instruction, seconds=4.0)
after = session.observe()
# 用 after 的图像和反馈检查动作条件，再选择下一条指令或修复步骤。
```

调用方已经持有图像和状态时，可以直接发给模型服务。

```python
observation = {
    'observation/image': global_rgb,
    'observation/wrist_image': wrist_rgb,
    'observation/state': state_13,
    'prompt': instruction,
}
actions = client.infer(observation)['actions']
# actions.shape == (50, 7)
```

可运行的离线输入示例使用 NPZ 中的三路同名数组。

```bash
bash run.sh examples/infer_observation.py \
  --sample /path/to/recorded-observation.npz \
  --instruction "完整的模型训练任务指令" \
  --output .runtime/veraplan/actions.npy
```

动作 7 维为基座系绝对 TCP `[x,y,z,rx,ry,rz]` 与夹爪命令。位置单位为米，姿态为旋转向量弧度，夹爪以 0.5 为开合阈值，1 表示打开。调用方获得动作数组后，可使用现有控制器执行；示例文件保存的是模型输出。

`instruction` 会原样作为语言条件传入。VeraPlan 根据当前画面和动作条件选择本次调用的指令。

## 运行最小闭环示例

先做只读观察，再使用真实图像做 dry-run，最后由操作者启动真机模式。

```bash
bash run.sh examples/veraplan_minimal.py --mode observe

bash run.sh examples/veraplan_minimal.py --mode dry-run --duration 4 \
  --prompt "完整的模型训练任务指令"

bash run.sh examples/veraplan_minimal.py --mode record \
  --execute --i-understand --home-at-start --duration 4 \
  --prompt "完整的模型训练任务指令"
```

已有 AGX 的独立服务可附加 `--port 8002 --ur5-repo /home/nvidia/ur5 --config /home/nvidia/ur5/config/ur5-orin.toml`。训练任务文字可从该 checkpoint 的 `source_model_spec.json` 读取 `prompt`，也可由模型所有者提供。

真机模式使用现有 HOME 约定 `[-180,-90,-90,-90,90,-90]` 度，启动 HOME 时同步请求夹爪全开。控制默认 30 Hz、每周期 1 mm / 1°，来自 `[inference]` 配置。一次 `act` 的时间由 `seconds` 决定，规划循环的轮数和总时长由 VeraPlan 管理。

## 暂停、停止与恢复

`act` 是同步调用。VeraPlan 可将它放入工作线程，并在规划线程中请求暂停或停止。

```python
from concurrent.futures import ThreadPoolExecutor
import time

with open_session(args) as session:
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(session.act, training_instruction, 4.0)
        while session.status()['execution_status'] == 'idle' and not job.done():
            time.sleep(0.01)
        # VeraPlan 收到暂停事件后执行以下操作。
        session.pause()
        status = session.wait(timeout=5)
        result = job.result()  # 推理、通信或 hold 错误在这里抛出。
        current = session.observe()
        # 后续 session.act(...) 使用新观察和新动作块继续执行。
```

- `pause()` 发出取消请求，活动状态为 `pausing`。控制循环随后调用现有 `robot.hold()`，关闭该段推理 worker。
- `wait()` 等待该段控制循环和 worker 退出，返回 `paused`、`completed`、`stopped` 或 `failed` 状态。请求发出后，调用方通过 `wait()` 和工作线程结果确认处理完成。
- `stop()` 请求终止当前会话，后续 `act()` 会报错。新动作需要重新创建会话。
- `with open_session(...)` 退出时停止活动段并关闭相机、串口和 RTDE 连接。CLI 的 Ctrl-C 经相同清理路径退出。

软件 hold 通过现有 URScript `CMD_IDLE` 和夹爪当前位置保持实现。夹爪保持使用已打开的串口连接。现场急停由设备硬件承担。

## 执行结束与任务成功

`act` 返回的结构如下。

```json
{
  "execution_status": "completed",
  "execution_done": true,
  "error": null,
  "task_success": null,
  "summary": {"ticks": 120, "sent": 120, "elapsed": 4.0, "dry_run": false}
}
```

`completed` 表示这次控制时间段已退出，hold 调用和 worker 清理已返回。模型每次产生 50 步动作，控制器持续按当前观察重规划，任务语义结束由 VeraPlan 结合新图像、反馈和验收条件判断。`task_success` 由上层判定后写入自己的实验记录。

`summary.sent` 统计实际下发周期，`dry_run` 表示执行模式。轨迹日志还包含 TCP、关节、夹爪读数、动作块序号、预测延迟与 waypoint 进度。`action_complete` 描述当前命令 waypoint 的执行进度，任务验收应使用视觉和实际结果。

例如当前条件式任务的检查项可以覆盖目标物是否可抓取、纸盒是否仍遮挡、物料是否已在透明盒内以及松爪后的最终状态。VeraPlan 侧维护这些条件和修复步骤。

## 可调用能力

- 读取 global、wrist RGB 画面和机器人、夹爪反馈。
- 按配置回到 HOME，启动时请求夹爪全开。
- 将语言指令和当前图像、状态传入 π0.5，执行一段模型动作。
- 暂停动作段并保持当前位置，从新观察发起后续动作。
- 停止会话，退出时释放相机、串口和 RTDE 连接。

VeraPlan 一侧在这些接口之上维护任务状态转换、动作条件检查、结果判断和修复步骤。模型权重由所有者单独提供。

## 日志与验证

最小示例的观察反馈输出到终端，动作日志写入 `--log-dir`。默认目录为 checkout 的 `.runtime/logs`。错误包含模型连接、相机新鲜度、RTDE 与夹爪故障，调用方保留异常及相关图像用于 VeraPlan 的恢复判断。

```bash
bash run.sh -m unittest discover -s tests -p test_planner_bridge.py -v
```

本次接口软件回归结果记录于 `docs/validation.md`。
