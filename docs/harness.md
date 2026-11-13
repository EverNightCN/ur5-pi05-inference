# HarnessVLA 使用说明

## 安装与私有配置

`bash setup_harness.sh` 创建独立 Python 3.11.14 环境，安装 `requirements/harness.lock`，再校验并解包固定版本 RPent。规划器通过 WebSocket 调用本机 π0.5 服务，CUDA 依赖位于独立的 Python 3.10 环境。

私有配置默认位于 `~/.config/harnessvla/ur5.env`，由 `HARNESSVLA_ENV_FILE` 指定其他路径。示例见 `config/harness.env.example`。

配置优先级为 CLI 参数、当前进程已有环境变量、私有配置文件。主要字段如下。

| 字段 | 用途 |
|---|---|
| `OPENAI_API_KEY` | 用户自行填写的 API 凭据 |
| `HARNESSVLA_MODEL` | RPent 的 provider-prefixed 模型 ID |
| `HARNESSVLA_BASE_URL` | 对应 API 地址 |
| `HARNESSVLA_REASONING_EFFORT` | 推理强度 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 当前进程使用的网络代理 |
| `NO_PROXY` | 直连地址，包含 localhost、127.0.0.1 和 ::1 |

当前示例为 `openai-responses:gpt-6-astra`，推理强度 `low`。模型支持图像输入，推理工具调用使用 Responses API。[OpenAI 官方模型文档](https://developers.openai.com/api/docs/models/gpt-6-astra)

```bash
bash run_harness.sh check
bash run_harness.sh dry-run
bash run_harness.sh run
```

`run` 会执行配置中的 HOME 并进入真机工具循环。Ctrl+C 取消当前动作、清理工具线程并关闭硬件连接。运行时间和模型请求轮数默认不限，通过 `--planner-timeout` 与 `--max-turns` 调整。

## 动作工具

| 工具 | 功能 |
|---|---|
| `observe` | 新鲜 global/wrist RGB 图像及 TCP、关节、夹爪实测状态 |
| `pi05_act` | 执行一段 VLA 动作，默认 4 秒，随后返回新观测 |
| `move_delta` | 基座系下的 TCP 平移，单位毫米 |
| `rotate_delta` | 基座系下的轴角旋转向量，单位度 |
| `set_gripper` | 打开或关闭夹爪，并确认动作完成反馈 |
| `list_dir` / `read_text_file` | 检索记忆和记录 |
| `write_text_file` / `finish` | 写入记录并提交观察结论 |

位姿调整沿用共享轨迹执行器，保持当前夹爪命令。规划器使用 `pi05_act` 时，默认沿用完整双物体训练指令。单物体子指令需要根据实际模型能力评估。

默认记忆位于 `.runtime/memory/ur5`，可通过 `--memory-dir` 或 `HARNESSVLA_MEMORY_DIR` 指定现有记忆目录。运行记录位于 `.runtime/runs/<时间戳>`，包含配置、图像、状态、动作日志、transcript 和 `result.json`。

`finish` 保存规划器的观察结论，`physical_success` 初始为 null，实际任务结果由真机实验验收。

## 使用 Mac v2rayN 代理

如果代理仅监听 Mac 的回环地址，可在 Mac 上通过 SSH 将其转发到 AGX。示例中的 SSH host alias `agx` 和本机代理端口根据实际情况设置。

```bash
# 在 Mac 上运行，保持此会话在线
ssh -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18080:127.0.0.1:10808 agx
```

在 AGX 的私有 Harness 配置中设置以下值。

```dotenv
HTTP_PROXY=http://127.0.0.1:18080
HTTPS_PROXY=http://127.0.0.1:18080
NO_PROXY=localhost,127.0.0.1,::1
```

这条链路依赖 Mac、v2rayN 和 SSH 会话在线。使用其他本地或局域网代理时，填写实际代理地址。机器人 RTDE/URScript 使用 TCP 直连，串口和相机通过本地设备访问。
