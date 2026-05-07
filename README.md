# FakeTrace

当前项目已经整理成适合继续扩展的取证工作台结构。

现在内置两类能力：

- `MARC`：图片真伪检测
- `Trufor`：篡改区域定位

## 推荐目录

```text
FakeTrace/
├─ configs/                    # 配置文件
├─ Inference/                  # 原始推理相关代码
├─ MARC/                       # MARC 方法相关代码
├─ Trufor/                     # TruFor 篡改定位相关代码
├─ src/
│  ├─ faketrace_app/           # FakeTrace 正式应用包入口
│  └─ marc_app/                # 兼容层，保留旧导入路径
│     ├─ api/                  # FastAPI 应用与路由
│     │  ├─ app.py
│     │  ├─ deps.py
│     │  └─ routes/
│     ├─ core/                 # 全局配置、路径、共享基础能力
│     ├─ features/             # 业务功能模块
│     │  ├─ detector/          # MARC 真伪检测
│     │  └─ trufor/            # TruFor 篡改定位
│     ├─ ui/                   # Web UI 资源
│     │  └─ assets/
│     │     └─ static/
│     ├─ cli.py                # CLI 入口逻辑
│     ├─ web.py                # 兼容层，保留旧入口
│     ├─ config.py             # 兼容层
│     ├─ inference_engine.py   # 兼容层
│     └─ paths.py              # 兼容层
├─ app.py                      # CLI 启动入口
└─ web_app.py                  # Web 启动入口
```

## 后续加功能的建议

- 新功能统一放到 `src/faketrace_app` 对应结构中，或先落到现有 `src/marc_app` 模块后再做兼容映射
- 每个功能的 HTTP 路由放到 `src/faketrace_app/api/routes/<feature_name>.py` 或兼容对应层
- 如果某个功能有独立前端资源，再放到 `src/faketrace_app/ui/assets/` 下拆目录
- `core/` 只放全局共用能力，不放具体业务逻辑

这样后面你要继续加批量任务、历史记录、报表导出、更多检测算法或别的取证模型时，都可以直接按模块扩展。
