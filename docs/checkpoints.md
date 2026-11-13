# 模型接入与转换

## 完整 checkpoint 目录

新 AGX 需要从模型持有者单独取得以下六个文件，放在同一目录。

```text
checkpoint/
├── model.safetensors
├── config.json
├── norm_stats.json
├── paligemma_tokenizer.model
├── source_model_spec.json
└── conversion_audit.json
```

`model.safetensors` 是合并后的完整 π0.5 权重。`norm_stats.json` 必须对应当前训练数据和模型，`paligemma_tokenizer.model` 提供本地 tokenizer。服务在加载期间使用该目录中的文件。

`config.json` 需要符合当前执行器约定。

```json
{
  "pi05": true,
  "action_dim": 32,
  "action_horizon": 50,
  "max_token_len": 200,
  "num_steps": 10,
  "paligemma_variant": "gemma_2b",
  "action_expert_variant": "gemma_300m"
}
```

`conversion_audit.json` 由转换器生成，记录全部十组 attention/FFN LoRA 合并、严格权重加载结果，以及上述五个模型文件的 SHA-256。attention 输出投影使用跨 head 求和的 LoRA 合并公式。接入工具会拒绝已知的旧版错误合并、缺失文件或内容变化。

```bash
python3 tools/configure_checkpoint.py /path/to/checkpoint
```

`.runtime/checkpoint-current` 为当前模型的唯一选择入口。切换模型后重启模型服务以加载新内容。

```bash
bash start_inference.sh restart
```

## 输入与动作语义

- `observation/image` 和 `observation/wrist_image` 为 HWC、RGB、uint8 图像，全局/腕部顺序必须对应训练时的顺序。
- `observation/state` 为 13 维浮点数组，依次为 TCP `[x,y,z,rx,ry,rz]`、六个关节角和夹爪开合状态。
- TCP 位置单位为米，TCP 旋转为基座坐标下的轴角旋转向量，关节角单位为弧度。
- 策略动作是绝对 TCP 位姿与二值夹爪命令，输出形状为 `(50,7)`。夹爪策略值 1 表示打开，0 表示闭合，与原始串口位置 1000/0 对应。
- 策略输入旋转使用训练数据的正 Ry 等价轴角分支。控制器通过 SO(3) 插值处理位姿，避免直接对等价旋转向量做线性插值。

任务工况的 HOME 为 `[-180,-90,-90,-90,90,-90]` 度。相机布置、TCP 和模型训练条件需要在新工位对应起来。global90 模型使用旋转后的全局机位，复现同一模型时需匹配其视觉条件。

## 从 JAX LoRA checkpoint 转换

已有审计后的 PyTorch checkpoint 时直接执行接入步骤。持有 JAX checkpoint 时，可使用仓库中的转换器。源目录需要包含 `params/`、`model_spec.json`、`norm_stats.json` 和 `paligemma_tokenizer.model`。

```bash
bash run.sh convert_checkpoint.py \
  --openpi "$PWD/vendor/openpi" \
  --checkpoint /path/to/jax-checkpoint \
  --output /path/to/new-pytorch-checkpoint
```

该步骤在 CPU 上恢复完整 JAX 参数并合并 LoRA，具有额外的内存和存储需求。转换器将拒绝覆盖已存在的输出目录。模型结构必须与训练时的规格一致，`model_spec.json` 应保存训练侧实际的 LoRA variant、维度和超参数。

需要量化验证 JAX/PyTorch 输出一致性时，使用训练侧生成的固定观察、固定 noise 和参考动作，再执行 `validate_reference.py`。

```bash
bash run.sh validate_reference.py \
  --checkpoint /path/to/new-pytorch-checkpoint \
  --references /path/to/reference-directory \
  --output .runtime/parity.json
```
