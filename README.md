# FakeTrace

FakeTrace 是一个面向多模态伪造内容检测的研究与工程化项目，当前整合了图像真伪检测、图像篡改定位、音频深伪检测、视频真伪检测，以及本地 CLI、Web UI 和 HTTP API 三种使用方式。

项目当前已经将主运行逻辑统一到 `src/faketrace_app/`，根目录入口文件仅保留为薄封装，便于后续继续扩展和维护。

## 主要能力

- 图像真伪检测：支持 `MARC`，并扩展支持 `Forensic-MoE`、`ForgeLens`、`LOTA`、`MF2DA`、`UnivFD`
- 图像篡改定位：支持 `TruFor`、`CAT-Net`、`Fassa`、`EffUnetPP`
- 音频深伪检测：支持基于 YAML 配置的训练、评估、阈值扫描和批量预测
- 视频真伪检测：支持 `TRI`
- 使用方式：支持 `CLI`、`Web UI`、`FastAPI`

## 项目特点

- 主代码集中在 `src/faketrace_app/`，结构更标准
- `models/` 下保留各模型代码和权重目录，便于自行管理大文件
- `configs/default.json` 统一管理主运行配置
- `docs/` 存放补充文档，`legacy/` 存放历史归档代码

## 安装依赖

先安装项目依赖：

```bash
pip install -r requirements.txt
```

如果你使用 GPU，请根据本机 CUDA 环境安装匹配版本的 `torch` 与 `torchvision`。

## 快速开始

### 1. 图像检测 CLI

单张图片：

```bash
python app.py --image path/to/image.jpg
```

指定 CPU：

```bash
python app.py --image path/to/image.jpg --device cpu
```

批量检测目录：

```bash
python app.py --image-dir path/to/images --recursive --batch-size 4
```

导出 JSON 和 CSV：

```bash
python app.py --image-dir path/to/images --save-json results.json --save-csv results.csv
```

### 2. Web UI
如果缺少对应权重，相关功能将无法正常加载。
模型权重链接： https://pan.baidu.com/s/1DD6ILIBk3Tq0ZjSzb8sQoQ?pwd=0123 提取码: 0123

启动本地 Web 服务：

```bash
python web_app.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

### 3. 音频实验 CLI

健康检查：

```bash
python audio_app.py audio-healthcheck ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_healthcheck
```

训练：

```bash
python audio_app.py audio-train ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --train-manifest data/manifests/train.csv ^
  --val-manifest data/manifests/val.csv ^
  --output-dir output/audio_ast ^
  --device cuda
```

评估：

```bash
python audio_app.py audio-eval ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_ast_eval ^
  --device cuda
```

批量预测：

```bash
python audio_app.py audio-predict ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --audio-dir data/eval_audio ^
  --output-dir output/audio_submission ^
  --fake-threshold 0.5 ^
  --save-probs
```

阈值扫描：

```bash
python audio_app.py audio-threshold-scan ^
  --config configs/audio/ast_audioset_ft.yaml ^
  --checkpoint output/audio_ast/best.pt ^
  --manifest data/manifests/val.csv ^
  --output-dir output/audio_thresholds ^
  --metric track2_macro_f1
```

## 当前目录结构

> 仓库默认不提交大体积模型权重文件，例如 `.pth`、`.pt`、`.tar`。  
> 读者请按照下面的目录结构自行放置权重。

权重地址：https://pan.baidu.com/s/1c0-7AAt7Az1lkAi_MGhs0g?pwd=0123 提取码: 0123 

欢迎邮箱联系：wpmu@sjtu.edu.cn

```text
FakeTrace/
├─ app.py
├─ audio_app.py
├─ web_app.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ configs/
│  ├─ default.json
│  └─ audio/
├─ docs/
│  ├─ cli_usage.md
│  ├─ project_structure.md
│  └─ *.md
├─ legacy/
│  ├─ README.md
│  └─ Inference/
├─ models/
│  ├─ marc/
│  ├─ trufor/
│  ├─ CAT-Net/
│  ├─ Fassa/
│  ├─ effunetpp/
│  ├─ TRI/
│  ├─ audio/
│  ├─ Forensic-MoE-main/
│  ├─ ForgeLens/
│  ├─ LOTA/
│  ├─ MF2DA/
│  └─ UnivFD/
├─ scripts/
└─ src/
   └─ faketrace_app/
      ├─ api/
      ├─ core/
      ├─ features/
      ├─ ui/
      ├─ cli.py
      ├─ audio_cli.py
      ├─ web.py
      ├─ config.py
      ├─ paths.py
      └─ inference_engine.py
```

## 权重文件放置说明

常用权重建议放置如下：

- `models/marc/pretrained/model_best.pth`
- `models/trufor/pretrained_models/trufor.pth.tar`
- `models/CAT-Net/output/best.pth.tar`
- `models/Fassa/fassa_best_model.pth`
- `models/effunetpp/effunetpp_best_model.pth`
- `models/audio/best.pt`

如果你启用了扩展图像检测模型，也建议将权重放在各自模型目录内部：

- `models/Forensic-MoE-main/...`
- `models/ForgeLens/...`
- `models/LOTA/...`
- `models/MF2DA/...`
- `models/UnivFD/...`

当前 `configs/default.json` 默认引用：

```json
{
  "checkpoint": "models/marc/pretrained/model_best.pth",
  "audio": {
    "checkpoint": "models/audio/best.pt"
  }
}
```

如果你修改了权重文件名或路径，请同步修改 `configs/default.json`，或者在命令行中通过 `--checkpoint` 覆盖。

## 支持的 API

### `GET /api/status`

返回当前服务中各模型的加载状态、设备、阈值和权重路径信息。

### `POST /api/predict`

图像真伪检测接口。

- 上传字段：`files`
- 可选参数：`model`
- 支持模型：`marc`、`forensic_moe`、`forgelens`、`lota`、`mf2da`、`univfd`

### `POST /api/localize`

图像篡改定位接口。

- 上传字段：`files`
- 可选参数：`model`、`save`、`output_dir`
- 支持模型：`trufor`、`catnet`、`fassa`、`effunetpp`

### `POST /api/audio/predict`

音频深伪检测接口。

- 上传字段：`files`

### `POST /api/video/predict`

视频真伪检测接口。

- 上传字段：`files`
- 可选参数：`model`
- 当前支持模型：`tri`

## 配置说明

主运行配置文件位于：

```text
configs/default.json
```

它主要控制：

- 图像主检测模型权重路径
- 音频主检测模型权重路径
- 设备类型，如 `auto`、`cpu`、`cuda:0`
- 批大小
- 输入尺寸
- 判定阈值

图像检测默认逻辑为：

```text
real_probability >= threshold  => real
否则                              => fake
```

## 代码组织说明

- `src/faketrace_app/api/`：FastAPI 应用和路由
- `src/faketrace_app/core/`：配置、路径等通用基础能力
- `src/faketrace_app/features/`：各检测与实验功能实现
- `src/faketrace_app/ui/`：前端页面与静态资源
- `models/`：第三方模型代码与权重目录
- `scripts/`：辅助脚本
- `legacy/`：历史代码归档，默认运行不依赖

## 相关文档

- [docs/cli_usage.md](./docs/cli_usage.md)
- [docs/project_structure.md](./docs/project_structure.md)
- [docs/audio_current_evidence.md](./docs/audio_current_evidence.md)
- [docs/audio_effectiveness_plan.md](./docs/audio_effectiveness_plan.md)
- [docs/audio_remote_workflow.md](./docs/audio_remote_workflow.md)
- [docs/audio_server_results_summary.md](./docs/audio_server_results_summary.md)

## 注意事项

- 首次运行前请先准备好对应模型权重
- `models/` 下包含多套研究代码，依赖环境可能并不完全一致
- 部分模型对 CUDA、PyTorch 版本和额外依赖较敏感
- 如果只使用主流程，建议优先从 `MARC`、`TruFor` 和音频主配置开始

## License

本项目采用 [LICENSE](./LICENSE) 中声明的授权方式。
