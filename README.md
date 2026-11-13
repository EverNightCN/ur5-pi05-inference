# UR5 π0.5 Inference on Jetson AGX Orin

面向 UR5、双 RGB 相机和 RS485 夹爪的完整 π0.5 真机推理部署，包含直接策略执行与 HarnessVLA 上层视觉规划两个入口。

```text
双相机 + RTDE 状态 → 本机 π0.5 WebSocket 服务 → 30 Hz 轨迹执行 → UR5 / 夹爪
                          ↑
             HarnessVLA / RPent 视觉规划与工具调用
```

## 功能

- 完整 π0.5 PyTorch 推理，BF16、10 步去噪、50 步动作块、7 维输出。
- SDPA、TorchInductor/Triton 编译与 CUDA Graph，优化 AGX Orin SM87 推理。
- 固定频率控制、后台异步推理、动作时间索引，以及位姿与夹爪共享的 waypoint 执行进度。
- HOME、只读观测、dry-run、真机 rollout、日志、checkpoint 审计和基准测试。
- HarnessVLA 使用 RPent 原生 API planner、图像工具回传、本地记忆和动作工具。

## 已验证环境

| 项目 | 版本或设备 |
|---|---|
| 计算设备 | NVIDIA Jetson AGX Orin 64GB，aarch64 / SM87 |
| 系统 | Ubuntu 22.04，JetPack 6.2 / L4T R36.4.3 |
| CUDA | 12.6 |
| 本机策略环境 | Python 3.10.12、Jetson Torch 2.11.0、Triton 3.6.0 |
| 规划器环境 | Python 3.11.14、RPent 固定版本、独立依赖锁文件 |
| 机械臂 | UR5 CB3，PolyScope 3.15.8 |
| 相机 | 两路 USB RGB，全局与腕部，1280×720 @ 30 FPS |
| 夹爪 | 大寰 PGC-300-60，Modbus RTU / RS485，115200，slave 1 |

安装脚本针对以上 JetPack/CUDA 组合校验环境。其他 JetPack 版本需要重新匹配 CUDA、Torch 和 Triton wheel。新设备应具有足够的本地存储来容纳 Python 环境、编译缓存和约 7.5 GB 的 checkpoint。

## 仓库结构

```text
run_robot.sh             直接 π0.5 真机入口
run_harness.sh           HarnessVLA 入口
start_inference.sh       按需管理模型服务
setup.sh                 安装本机 CUDA 推理环境
setup_harness.sh         安装独立的 API 规划环境
runtime.py / serve.py    完整模型加载与 WebSocket 服务
policy_rollout.py        配置、硬件连接、HOME 与控制循环
rollout_io.py            新鲜帧、后台推理与共享轨迹执行器
harness_*.py             RPent 规划器入口及 UR5 工具
config/                  设备与规划器配置示例
requirements/            依赖锁文件、公开安装包 URL 和 SHA-256
tools/                   环境检查、模型接入、预热与服务安装
ur5_teleop/               RTDE、夹爪、HOME 与旋转数学支持
urscript/                UR5 机器人端程序
vendor/                  实测 OpenPI 源码快照、官方 RTDE 客户端
tests/                   使用模拟硬件的回归测试
docs/                    模型接入、HarnessVLA 与复现验证说明
```

运行时文件写入当前 checkout 的 `.runtime/`，环境写入 `.venv/` 和 `.venv-harness/`。部署路径由脚本位置解析，可克隆到不同用户和存储盘。

## 1. 安装

先为 AGX 安装对应 JetPack，并确保 CUDA 12.6 开发文件可用。

```bash
git clone https://github.com/EverNightCN/ur5-pi05-inference.git
cd ur5-pi05-inference
sudo apt-get update
sudo apt-get install -y python3.10-dev python3-pip build-essential patch curl v4l-utils
python3 -m pip install --user uv==0.9.9
bash setup.sh
```

`setup.sh` 从公开地址获取 Jetson Torch、Triton 和 NVIDIA cuDSS，逐一校验 SHA-256，安装固定 Python 依赖，再应用当前仓库的 OpenPI 补丁和 Transformers 替换文件。最后实际执行 CUDA BF16、Triton 编译和推理模块导入检查。

网络环境可通过 `HTTP_PROXY`、`HTTPS_PROXY` 和 uv 的 `UV_INDEX_URL` 设置。已有离线安装包时，指定其所在目录，脚本仍会核对内容。

```bash
PI05_ASSETS_DIR=/path/to/offline-assets bash setup.sh
```

三个软件安装包的文件名和校验值见 `requirements/assets.json`。它们由 NVIDIA/Jetson 软件分发站点获取。

## 2. 配置硬件

安装脚本首次创建 `config/ur5.local.toml`。按新工位修改以下字段。

| 配置项 | 内容 |
|---|---|
| `[robot].ip` | UR5 控制器 IP |
| `[collect].camera_indices` | 按 global、wrist 顺序填写 V4L2 设备路径 |
| `[gripper].port` | USB-RS485 串口路径 |
| `[collect].task_en` | 与模型训练语言条件对应的任务 |
| `[safety].workspace_min/max` | 当前控制器使用的基座坐标工作区域 |

相机可通过 `v4l2-ctl --list-devices` 枚举。优先使用 `/dev/v4l/by-id/` 路径来保持相机顺序稳定。当前用户需要相机和串口的访问权限。

```bash
sudo usermod -aG video,dialout "$USER"
```

重新登录后权限生效。机械臂控制器需要允许来自 AGX 的 RTDE、Dashboard 和 URScript 连接，使用端口 30004、29999 和 30001。

默认控制参数为 30 Hz、动作时间轴 30 FPS、推理间隔 0.4 秒、每周期位移 1 mm、转角 1°。默认运行时间为 0，即持续到 Ctrl+C。该任务 HOME 为 `[-180, -90, -90, -90, 90, -90]` 度，入口会核对该约定。

## 3. 接入模型

将已转换并审计的完整 checkpoint 单独拷贝到新 AGX，例如 `/path/to/checkpoint`。接入所需的六个文件及 JAX 转换方法见 [模型接入说明](docs/checkpoints.md)。

```bash
python3 tools/configure_checkpoint.py /path/to/checkpoint
bash run.sh tools/make_warmup_sample.py
python3 tools/install_service.py
bash start_inference.sh start
bash start_inference.sh status
```

模型接入工具检查十组 LoRA 合并记录、模型规格以及全部模型文件的 SHA-256，再创建 `.runtime/checkpoint-current`。预热输入由脚本生成，供模型加载与编译使用。

服务名称为 `ur5-pi05-inference.service`，通过当前用户的 systemd 按需启动。服务安装操作执行 unit 写入及 `daemon-reload`，启动由 `start_inference.sh start` 显式触发。首次加载和编译默认等待 600 秒，可通过 `--timeout` 调整。

## 4. 直接 π0.5 推理

```bash
# 只读硬件观测
bash run_robot.sh observe

# 使用真实观测推理并记录动作预测
bash run_robot.sh dry-run --duration 15

# 执行 HOME，然后持续执行模型动作
bash run_robot.sh run

# 或指定运行时间
bash run_robot.sh run --duration 60
```

`run` 和 `home` 会产生机械臂运动。`run` 启动模型服务，使用同一 RTDE 连接执行 HOME 和伺服初始化，然后进入 rollout。Ctrl+C 请求停止并清理硬件连接。测试时保持可操作急停，并按当前工位确认 HOME 运动路径。

可通过 `--config /path/to/config.toml` 选择其他设备配置，同名命令行参数覆盖 `[inference]` 配置。完整选项通过 `bash run_robot.sh run --help` 查看。

```bash
# 单独回 HOME
bash run_robot.sh home

# 释放模型服务占用的内存
bash start_inference.sh stop
```

每轮直接推理日志位于 `.runtime/logs/`，模型服务日志为 `.runtime/logs/service.log`。

## 5. HarnessVLA 推理

HarnessVLA 在本地 π0.5 之上增加视觉规划。规划器经 API 观察双摄像头图像，调用短段 VLA、位姿或夹爪原语，再获取新观测。

```bash
bash setup_harness.sh
mkdir -p ~/.config/harnessvla
chmod 700 ~/.config/harnessvla
install -m 600 config/harness.env.example ~/.config/harnessvla/ur5.env
# 在本机编辑 ~/.config/harnessvla/ur5.env，填写自己的凭据
bash run_harness.sh check
bash run_harness.sh dry-run
bash run_harness.sh run
```

配置示例采用 `openai-responses:gpt-6-astra` 和 `low` 推理强度。模型、API 地址和推理强度均由私有配置或 CLI 参数设置。`check` 检查依赖与配置读取，实际 API 和硬件链路由 `dry-run` 验证。

工具、记忆、代理配置与结果格式详见 [HarnessVLA 使用说明](docs/harness.md)。

## 6. 测试与性能复现

```bash
bash run.sh -m unittest discover -s tests -v
bash run.sh -m unittest test_checkpoint test_convert_checkpoint -v
bash run_harness.sh check
```

安装 Harness 环境后，在同一环境运行其原生工具回归测试。

```bash
PYTHONPATH="$PWD:$PWD/.runtime/RPent:$PWD/vendor/openpi/packages/openpi-client/src" \
  .venv-harness/bin/python -m unittest discover -s tests -p test_harness_vla.py -v
```

同一 checkpoint 的基准测试使用固定观察输入，并明确开启与生产服务相同的优化。

```bash
bash run.sh benchmark.py \
  --checkpoint .runtime/checkpoint-current \
  --sample /path/to/recorded-observation.npz \
  --attention sdpa --compile-mode default --compile-all \
  --cuda-graph --cache-padding --iterations 30 \
  --output .runtime/benchmark.json
```

既有 global90 checkpoint 的 AGX 服务延迟 P50/P95 为约 420.33/422.48 ms，10 步去噪、50×7 输出。新设备的功耗档位、温度和其他负载会影响结果，使用同一 checkpoint 与观测重新测量。功耗和频率由设备当前任务策略管理。

复现验证的实际执行范围及当前限制见 [验证记录](docs/validation.md)。

## 常见问题

- **模型路径失效** — 检查外置盘挂载及 `.runtime/checkpoint-current`。接入工具校验的六个文件需要保持完整。
- **相机打开失败** — 检查 V4L2 节点、权限和其他进程是否占用。
- **CUDA/Triton 导入失败** — 对照 JetPack/CUDA 版本，重新运行 `bash setup.sh`，读取具体检查错误。
- **首次服务等待超时** — 查看 `.runtime/logs/service.log`，完成问题诊断后通过 `--timeout 1200` 扩展编译等待时间。
- **新工位任务成功率变化** — 重新核对相机位置、全局/腕部顺序、TCP、夹爪语义、训练归一化统计和任务指令。代码部署成功后仍需完成对应模型的真机任务验收。

## 代码来源

本机推理由 [OpenPI](https://github.com/Physical-Intelligence/openpi) 实测源码快照和 Jetson 补丁构成。上层规划使用 [RPent / HarnessVLA](https://github.com/RLinf/RPent) 固定版本。控制循环结构参考 [LeRobot](https://github.com/huggingface/lerobot)，硬件支持代码来源于 [UR5 Teleop](https://github.com/HDU-EILab/ur5-teleop)。第三方来源、版本及许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 许可证

本项目代码使用 MIT License。OpenPI 使用 Apache-2.0，官方 RTDE Python 客户端使用 BSD-3-Clause。第三方源码保留各自许可。
