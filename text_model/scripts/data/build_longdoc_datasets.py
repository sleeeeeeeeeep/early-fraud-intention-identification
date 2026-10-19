#!/usr/bin/env python3
"""兼容入口：构建 V361 Luna 子代理长文本任务包。

当前路线从零生成完整对话；旧版拼接/扩写流程已经退出正式执行路径。
"""

try:
    from text_model.scripts.data.build_longdoc_subagent_tasks import main
except ModuleNotFoundError:
    from build_longdoc_subagent_tasks import main


if __name__ == "__main__":
    raise SystemExit(main())
