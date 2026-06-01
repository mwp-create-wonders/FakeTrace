# FakeTrace

> 一个面向图像伪造检测与篡改区域定位的中文工作台，集成 `MARC`、`TruFor`、`CAT-Net`、`Fassa`、`EffUnetPP` 多种能力，支持 `CLI`、`Web UI` 和 `HTTP API` 三种使用方式。

## 项目简介

FakeTrace 是一个将多种图像取证模型统一封装后的应用工程，目标是把“真假判别”和“篡改区域定位”放进同一套可运行、可扩展、可演示的工作流里。

你可以用它来做这些事：

- 判断一张图片更接近真实图像还是 AI / 伪造图像
- 对疑似篡改区域做像素级热力图定位
- 通过 Web 页面上传图片并直接查看可视化结果
- 通过 CLI 批量跑文件夹并导出 `JSON / CSV`
- 通过 API 把检测能力接入你自己的前后端系统

---

## 功能总览

### 1. 真伪检测

- 使用 `MARC` 模型进行 `real / fake` 二分类
- 返回 `real_probability`、`fake_probability` 和最终预测标签
- 支持单张、多张、文件夹、递归目录批量推理
- 支持阈值、设备、批大小、输入尺寸等运行参数覆盖

### 2. 篡改区域定位

当前已统一接入 4 个定位模型：

- `TruFor`
- `CAT-Net`
- `Fassa`
- `EffUnetPP`

定位接口可返回：

- 篡改热力图 `localization_map`
- 原图叠加图 `overlay`
- 部分模型的置信图 `confidence_map`
- 可疑区域占比 `suspicious_ratio`
- 部分模型的整体分数 `score`

### 3. Web 可视化工作台

- 拖拽上传图片
- 检测 / 定位双模式切换
- 定位模式下可切换不同模型
- 实时展示状态、预测结果、热力图和叠加图
- 适合演示、验收、人工分析和快速对比

### 4. 命令行批处理

- 支持单图推理
- 支持目录批量推理
- 支持递归扫描
- 支持结果导出为结构化文件

### 5. HTTP API

- `GET /api/status` 查看服务和模型状态
- `POST /api/predict` 执行真伪检测
- `POST /api/localize` 执行篡改定位

---

## 模型能力矩阵

| 能力 | 模型 | 输出类型 | 说明 |
| --- | --- | --- | --- |
| 真伪检测 | `MARC` | 分类结果 + 概率 | 判断图片更接近 `real` 还是 `fake` |
| 篡改定位 | `TruFor` | 热力图 + 叠加图 + 置信图 | 支持较完整的定位可视化输出 |
| 篡改定位 | `CAT-Net` | 热力图 + 叠加图 | 适合 JPEG / 压缩痕迹相关分析 |
| 篡改定位 | `Fassa` | 热力图 + 叠加图 | 使用特征增强方式做分割定位 |
| 篡改定位 | `EffUnetPP` | 热力图 + 叠加图 + 分数 | 轻量一些，适合快速接入 |

---

## 项目亮点

- 一个入口整合多种图像取证模型，不需要为每个模型单独搭环境和写调用逻辑
- `CLI + Web + API` 三套使用方式同时具备，适合研究、演示、部署接入
- 模型实现与应用代码分层清晰，便于后续继续扩展新模型
- 默认配置集中在 `configs/default.json`，便于统一调参
- Web 页面对检测与定位结果做了统一的可视化封装，更适合中文场景展示

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

当前根依赖文件中包含的核心包有：

- `torch`
- `torchvision`
- `pillow`
- `numpy`
- `fastapi`
- `uvicorn[standard]`
- `python-multipart`

说明：

- 部分定位模型依赖其各自目录下的额外实现与权重文件
- 如果你要完整使用全部模型，请确保 `models/` 下对应模型目录、配置和权重文件已经就位
- 如果你使用的是 Windows + CUDA，请安装与本机 CUDA 版本兼容的 `torch / torchvision`

### 2. 准备模型权重

项目默认会从以下位置读取模型文件：

- `models/marc/pretrained/model_best.pth`
- `models/trufor/pretrained_models/trufor.pth.tar`
- `models/CAT-Net/output/best.pth.tar`
- `models/Fassa/fassa_best_model.pth`
- `models/effunetpp/effunetpp_best_model.pth`

如果缺少对应权重，相关功能将无法正常加载。

### 3. 启动 Web 应用

```bash
python web_app.py
```

启动后访问：

```text
http://127.0.0.1:7860
```

### 4. 运行命令行检测

单张图片：

```bash
python app.py --image path/to/image.jpg
```

指定设备：

```bash
python app.py --image path/to/image.jpg --device cpu
```

批量检测目录：

```bash
python app.py --image-dir path/to/images --recursive --batch-size 4
```

导出结果：

```bash
python app.py --image-dir path/to/images --save-json results.json --save-csv results.csv
```

---

## CLI 使用说明

### 输入方式

二选一：

- `--image` 传入单张或多张图片，可重复传参
- `--image-dir` 传入图片目录

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 指定配置文件，默认 `configs/default.json` |
| `--checkpoint` | 覆盖检测模型权重路径 |
| `--device` | 指定设备，如 `auto`、`cpu`、`cuda`、`cuda:0` |
| `--recursive` | 递归扫描目录 |
| `--batch-size` | 覆盖推理批大小 |
| `--image-size` | 覆盖输入尺寸 |
| `--threshold` | 覆盖真假判断阈值 |
| `--save-json` | 保存 JSON 结果 |
| `--save-csv` | 保存 CSV 结果 |

### 默认判定规则

FakeTrace 默认使用以下规则输出标签：

```text
real_probability >= threshold  => real
否则                           => fake
```

默认阈值来自 `configs/default.json`，当前为 `0.5`。

---

## Web 页面能力

Web 工作台目前提供两种模式：

### 检测模式

- 上传一张或多张图片
- 返回每张图的 `Real / Fake` 结果
- 显示真假概率和置信条

### 定位模式

- 上传一张或多张图片
- 选择定位模型
- 展示原图、热力图、叠加图
- 对支持的模型显示额外置信图与分数

这套界面很适合：

- 给老师 / 团队演示
- 快速检查模型输出是否合理
- 对比不同定位模型的输出风格

---

## API 说明

### `GET /api/status`

查看当前服务状态，以及检测模型和定位模型是否成功加载。

返回内容包括：

- 服务是否就绪
- 检测模型设备、阈值、输入尺寸、批大小、权重路径
- `TruFor` 运行设备、实验配置、模型文件

### `POST /api/predict`

上传图片并执行真伪检测。

请求：

- `multipart/form-data`
- 字段名：`files`

返回：

- `results`
- `meta.device`
- `meta.threshold`
- `meta.checkpoint`

### `POST /api/localize`

上传图片并执行篡改区域定位。

请求参数：

- 文件字段：`files`
- 查询参数：`model`
- 查询参数：`save`
- 查询参数：`output_dir`

支持的 `model`：

- `trufor`
- `catnet`
- `fassa`
- `effunetpp`

返回内容通常包括：

- 图片文件名
- `suspicious_ratio`
- `localization_map_url`
- `overlay_url`
- 某些模型的 `score`
- 某些模型的 `confidence_map_url`

---

## 配置说明

默认运行配置位于：

```text
configs/default.json
```

当前默认配置包含：

- 检测模型权重路径
- 设备选择
- 批大小
- 输入尺寸
- 判定阈值
- MARC 模型结构参数

示例：

```json
{
  "checkpoint": "models/marc/pretrained/model_best.pth",
  "device": "auto",
  "batch_size": 1,
  "image_size": 336,
  "threshold": 0.5
}
```

---

## 项目结构

```text
FakeTrace/
├─ app.py                          # CLI 入口
├─ web_app.py                      # Web 服务入口
├─ configs/
│  └─ default.json                 # 默认运行配置
├─ models/                         # 外部模型实现与权重目录
│  ├─ marc/
│  ├─ trufor/
│  ├─ CAT-Net/
│  ├─ Fassa/
│  └─ effunetpp/
├─ src/
│  └─ faketrace_app/
│     ├─ api/                      # FastAPI 应用与路由
│     ├─ core/                     # 配置与路径管理
│     ├─ features/                 # 各能力封装
│     │  ├─ detector/
│     │  ├─ trufor/
│     │  ├─ catnet/
│     │  ├─ fassa/
│     │  └─ effunetpp/
│     ├─ ui/                       # Web 静态资源
│     ├─ cli.py                    # CLI 逻辑
│     ├─ config.py                 # 配置导出
│     ├─ inference_engine.py       # 推理导出
│     ├─ paths.py                  # 路径导出
│     └─ web.py                    # Web 导出
├─ CLI_USAGE.md
└─ requirements.txt
```

---

## 适用场景

- 图像伪造检测课程项目
- 图像取证 / AI 生成内容识别实验平台
- 模型集成展示与答辩演示
- 后续论文复现、接口封装和二次开发

---

## 扩展建议

如果你准备继续完善这个项目，后续很适合往这些方向扩展：

- 增加更多中文说明和论文链接
- 为每个模型补充独立的安装说明
- 增加批量定位 CLI
- 增加结果保存、历史记录和对比视图
- 增加 Docker 部署方案
- 增加 Swagger 示例和前后端联调用例

---

## License

项目根目录包含 [LICENSE](./LICENSE)。

## 相关算法贡献说明
本项目参考了：https://grip-unina.github.io/TruFor，https://github.com/mjkwon2021/cat-net以及https://github.com/edocipriano/fassa这些优秀的开源项目，在此感谢！
