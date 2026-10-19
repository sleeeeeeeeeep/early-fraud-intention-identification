#!/usr/bin/env python3
from __future__ import annotations
"""
TeleFraud 电信诈骗检测 - Gradio 网页 Web 服务与命令行检测工具

启动方式：
    python run.py                     # 启动极简 Web 界面
    python run.py [音频文件或文件夹路径]   # 命令行模式直接检测并输出结果
"""

# 模型路径与输出目录与缓存目录（相对于项目根目录）
ASR_MODEL_PATH = "models/qwen3_asr_1_7b_hf"
TRANSLATOR_MODEL_PATH = "models/qwen3_1_7b"
CLS_MODEL_PATH = "models/m1_e1b"
AUDIO_BASE_PATH = "models/qwen25_omni_7b"
AUDIO_CHECKPOINT_PATH = "models/audio_v334c"
OUTPUT_DIR     = "outputs"
CACHE_DIR      = "caches"

import argparse
import sys
import csv
import io
import time
from pathlib import Path

# Gradio上传大型音频批次时会同时持有多个临时文件。Codex/GUI启动的
# macOS进程可能只有256个文件句柄，启动时主动提升，避免Errno 24。
try:
    import resource

    _soft_nofile, _hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    _target_nofile = (
        8192
        if _hard_nofile == resource.RLIM_INFINITY
        else min(8192, _hard_nofile)
    )
    if _soft_nofile < _target_nofile:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (_target_nofile, _hard_nofile),
        )
except (ImportError, OSError, ValueError):
    pass

import gradio as gr

from predict import TeleFraudDetector, LABEL_NAMES

FRAUD_COLOR   = "#d32f2f"
SAFE_COLOR    = "#2e7d32"
UNKNOWN_COLOR = "#e65100"

# 全局 detector，启动时加载
_detector: TeleFraudDetector | None = None


def make_label_bar(label_name: str, is_fraud: bool | None) -> str:
    if is_fraud:
        bg_color = "#fef2f2"
        border_color = "#fca5a5"
        text_color = "#b91c1c"
        tag = "⚠️ 检测到涉诈嫌疑"
    elif is_fraud is False:
        bg_color = "#f0fdf4"
        border_color = "#86efac"
        text_color = "#15803d"
        tag = "✅ 检测安全"
    else:
        bg_color = "#fffbeb"
        border_color = "#fcd34d"
        text_color = "#b45309"
        tag = "❓ 未知场景"
        
    return f"""
<div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 16px; font-family: system-ui, -apple-system, sans-serif; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
  <div style="display: flex; align-items: center; margin-bottom: 8px;">
    <span style="font-size: 16px; font-weight: 700; color: {text_color};">{tag}</span>
  </div>
  <div style="font-size: 14px; color: #475569; margin-bottom: 10px;">
    诈骗子类 analysis：<b style="color: #0f172a; font-size: 15px;">{label_name}</b>
  </div>
</div>
"""


# ── 推理函数 ──────────────────────────────────────────────────────────────────

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}


def _run_batch(paths: list[str], progress=None):
    """批量推理生成器，逐条产生结果以支持实时渲染，并在最后一条生成 CSV 文件路径。"""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    
    local_paths = []
    for p in paths:
        dest = cache_dir / Path(p).name
        try:
            shutil.copy(p, dest)
            local_paths.append(str(dest))
        except Exception:
            local_paths.append(p)

    total = len(local_paths)
    results = []
    
    for i, path in enumerate(local_paths):
        t0 = time.time()
        if progress:
            progress((i + 0.1) / total, desc=f"正在处理 {i+1}/{total}...")
        r = _detector.predict(path)
        r["elapsed_s"] = round(time.time() - t0, 2)
        r["audio_path"] = path
        results.append(r)
        
        if i + 1 == total:
            # 写入 CSV
            out_dir = Path(OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path  = str(out_dir / f"batch_results_{timestamp}.csv")
            with open(csv_path, "w", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["audio_path", "label_id", "label_name", "is_fraud",
                                "confidence", "route", "elapsed_s", "detected_language",
                                "source_transcript", "transcript"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(results)
            yield results, csv_path
        else:
            yield results, None


def on_files_change(files):
    """当上传文件变化时，如果为空则自动隐藏结果面板并禁用检测按钮；如果不为空则激活检测按钮。"""
    if not files:
        return gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False)
    return gr.update(), gr.update(), gr.update(interactive=True)


def predict_unified(files, progress=gr.Progress()):
    """统一检测入口，采用生成器实现瞬时流式UI渲染与实时展示，防止二次点击与加载卡死。"""
    # 1. 瞬间禁用按钮，重置并隐藏下方的输出面板
    yield (
        gr.update(interactive=False, value="正在检测中..."),
        gr.update(visible=False),
        gr.update(visible=False)
    )

    try:
        if not files:
            yield (
                gr.update(interactive=True, value="开始检测"),
                gr.update(visible=True, value="<div style='background-color:#fee2e2; border:1px solid #fca5a5; padding:12px; border-radius:8px; color:#b91c1c; font-weight:600;'>❌ 请先上传音频文件或选择文件夹！</div>"),
                gr.update(visible=False)
            )
            return

        print("=" * 50)
        print(f"收到上传内容，总共 {len(files)} 个上传项。")
        
        paths = []
        for f in files:
            if isinstance(f, dict):
                file_path = f.get("path") or f.get("name")
            else:
                file_path = getattr(f, "path", None) or getattr(f, "name", None)
                
            if not file_path:
                continue
                
            p = Path(file_path)
            if p.exists() and p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in AUDIO_EXTS:
                        paths.append(str(sub))
            elif p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                paths.append(str(p))
                
        paths = sorted(list(set(paths)))
        print(f"  解析过滤后，最终获得待检测音频文件 {len(paths)} 个")
        print("=" * 50)

        if not paths:
            error_msg = """
            <div style="background-color: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px; padding: 16px 20px; font-family: system-ui, -apple-system, sans-serif; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
              <h4 style="margin: 0 0 6px 0; color: #b45309; font-size: 15px; font-weight: 700;">⚠️ 未识别到支持的音频文件</h4>
              <p style="margin: 0; color: #78350f; font-size: 13px; line-height: 1.5;">
                如果您是直接拖拽了<b>文件夹</b>，受浏览器沙箱安全限制，浏览器可能无法递归读取其中的文件（显示为 0.0 B）。<br/>
                <b>解决方法：</b><br/>
                1. <b>点击框体</b>，在弹出的系统对话框中选中该文件夹上传；<br/>
                2. 或者打开该文件夹，<b>选中内部所有的音频文件</b>，直接拖拽到框内上传。
              </p>
            </div>
            """
            yield (
                gr.update(interactive=True, value="开始检测"),
                gr.update(visible=True, value=error_msg),
                gr.update(visible=False)
            )
            return

        total = len(paths)
        batch_start = time.time()
        
        # 3. 运行流式批处理并逐条渲染
        for results, csv_path in _run_batch(paths, progress):
            rows_html = ""
            for r in results:
                name  = Path(r["audio_path"]).name
                preview = (r["transcript"] or "")[:80]
                
                if r["is_fraud"]:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #fee2e2; color: #dc2626; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #fecaca; box-shadow: 0 1px 2px rgba(220, 38, 38, 0.05);">涉诈</span>'
                elif r["is_fraud"] is False:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #dcfce7; color: #16a34a; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #bbf7d0; box-shadow: 0 1px 2px rgba(22, 163, 74, 0.05);">安全</span>'
                else:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #fef3c7; color: #d97706; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #fde68a; box-shadow: 0 1px 2px rgba(217, 119, 6, 0.05);">未知</span>'

            rows_html = ""
            for idx, r in enumerate(results):
                name  = Path(r["audio_path"]).name
                preview = (r["transcript"] or "")[:80]
                
                # 为最新的行加上流式上屏淡入与高亮动画
                row_class = 'class="new-row" ' if (idx == len(results) - 1) else ''
                
                if r["is_fraud"]:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #fee2e2; color: #dc2626; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #fecaca; box-shadow: 0 1px 2px rgba(220, 38, 38, 0.05);">涉诈</span>'
                elif r["is_fraud"] is False:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #dcfce7; color: #16a34a; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #bbf7d0; box-shadow: 0 1px 2px rgba(22, 163, 74, 0.05);">安全</span>'
                else:
                    pill = '<span style="display:inline-block; padding: 4px 10px; background-color: #fef3c7; color: #d97706; border-radius: 9999px; font-weight: 600; font-size: 12px; border: 1px solid #fde68a; box-shadow: 0 1px 2px rgba(217, 119, 6, 0.05);">未知</span>'

                rows_html += f"""
                <tr {row_class}style="border-bottom: 1px solid #f1f5f9; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='#f8fafc'" onmouseout="this.style.backgroundColor='transparent'">
                  <td style="padding: 14px 16px; font-size: 14px; font-weight: 600; color: #1e293b;">{name}</td>
                  <td style="padding: 14px 16px; text-align: center;">{pill}</td>
                  <td style="padding: 14px 16px; font-size: 14px; color: #334155; font-weight: 500;">{r['label_name']}</td>
                  <td style="padding: 14px 16px; font-size: 14px; color: #64748b; font-family: monospace; text-align: right;">{r['elapsed_s']}s</td>
                  <td style="padding: 14px 16px; font-size: 13px; color: #64748b; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{preview}</td>
                </tr>"""

            fraud_count = sum(1 for r in results if r["is_fraud"])
            safe_count  = sum(1 for r in results if r["is_fraud"] is False)
            total_elapsed = time.time() - batch_start
            avg_elapsed   = total_elapsed / len(results) if results else 0

            # 滚动容器 + 固定粘性表头高级设计 + 渐现动画类 fade-in-panel
            table_html = f"""
            <div class="fade-in-panel">
              <div style="max-height: 600px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02); background: #ffffff; scrollbar-width: thin;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: system-ui, -apple-system, sans-serif;">
                  <thead style="position: sticky; top: 0; background: #f8fafc; z-index: 10; box-shadow: 0 1px 0 #e2e8f0;">
                    <tr>
                      <th style="padding: 14px 16px; font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">文件名</th>
                      <th style="padding: 14px 16px; font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;">结果</th>
                      <th style="padding: 14px 16px; font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">分类</th>
                      <th style="padding: 14px 16px; font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; text-align: right;">耗时</th>
                      <th style="padding: 14px 16px; font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">转写预览</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
              </div>
              
              <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; font-family: system-ui, -apple-system, sans-serif;">
                <span style="display: inline-flex; align-items: center; background: #f1f5f9; color: #475569; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; border: 1px solid #e2e8f0;">
                  📂 已完成 <b>{len(results)}</b>/{total} 条音频
                </span>
                <span style="display: inline-flex; align-items: center; background: #fee2e2; color: #b91c1c; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; border: 1px solid #fca5a5;">
                  🚨 涉诈 <b>{fraud_count}</b> 条
                </span>
                <span style="display: inline-flex; align-items: center; background: #dcfce7; color: #15803d; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; border: 1px solid #86efac;">
                  ✅ 安全 <b>{safe_count}</b> 条
                </span>
                <span style="display: inline-flex; align-items: center; background: #e0e7ff; color: #4338ca; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; border: 1px solid #c7d2fe; margin-left: auto;">
                  ⏱️ 累计耗时 <b>{total_elapsed:.2f}s</b> &nbsp;|&nbsp; 平均 <b>{avg_elapsed:.2f}s</b>/条
                </span>
              </div>
            </div>
            """

            if total == 1:
                r = results[0]
                label_bar = make_label_bar(r["label_name"], r["is_fraud"])
                label_bar += f'<p style="color:#999;font-size:12px;margin-top:4px">耗时 {r["elapsed_s"]:.1f}s</p>'
                combined_html = label_bar + "<hr style='margin:16px 0; border:0; border-top:1px solid #eee;'/>" + table_html
            else:
                combined_html = table_html

            if csv_path:
                yield (
                    gr.update(interactive=True, value="开始检测"),
                    gr.update(visible=True, value=combined_html),
                    gr.update(visible=True, value=csv_path)
                )
            else:
                yield (
                    gr.update(interactive=False, value=f"检测中... ({len(results)}/{total})"),
                    gr.update(visible=True, value=combined_html),
                    gr.update(visible=False)
                )
            
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"检测过程中发生异常:\n{err_detail}")
        error_card = f"""
        <div style="background-color: #fee2e2; border: 1px solid #fca5a5; border-radius: 12px; padding: 16px 20px; font-family: system-ui, -apple-system, sans-serif;">
          <h4 style="margin: 0 0 6px 0; color: #b91c1c; font-size: 15px; font-weight: 700;">❌ 检测出错</h4>
          <p style="margin: 0; color: #991b1b; font-size: 13px; font-family: monospace; white-space: pre-wrap;">{str(e)}</p>
        </div>
        """
        yield (
            gr.update(interactive=True, value="开始检测"),
            gr.update(visible=True, value=error_card),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=None)
        )


# ── 构建界面 ──────────────────────────────────────────────────────────────────

def build_ui(model_info: str) -> gr.Blocks:
    with gr.Blocks(
        title="电信诈骗检测系统",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .container-box {
            max-width: 1400px;
            margin: 0 auto !important;
            padding: 10px;
        }
        .header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .header h1 {
            font-size: 24px;
            font-weight: 800;
            margin: 0;
            color: white;
            letter-spacing: -0.5px;
        }
        .header p {
            font-size: 13px;
            margin: 6px 0 0;
            color: #93c5fd;
            font-weight: 500;
        }
        .status-bar {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 13px;
            color: #047857;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }
        .gradio-container button.primary {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            color: white !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25), 0 2px 4px -1px rgba(37, 99, 235, 0.15) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer;
        }
        .gradio-container button.primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3), 0 4px 6px -2px rgba(37, 99, 235, 0.2) !important;
            filter: brightness(1.05);
        }
        .gradio-container button.primary:active {
            transform: translateY(1px);
            box-shadow: 0 2px 4px -1px rgba(37, 99, 235, 0.2) !important;
        }
        
        /* 切换与流式上屏的淡入、平移高级动画 */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .fade-in-panel {
            animation: fadeInUp 0.45s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        
        @keyframes rowFadeIn {
            from {
                opacity: 0;
                background-color: #eff6ff;
            }
            to {
                opacity: 1;
                background-color: transparent;
            }
        }
        .new-row {
            animation: rowFadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        .dir-upload-btn {
            margin-top: 8px !important;
            margin-bottom: 12px !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #475569 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 10px 16px !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            cursor: pointer;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
        }
        .dir-upload-btn:hover {
            background: #f1f5f9 !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
        }
        """
    ) as demo:

        with gr.Column(elem_classes="container-box"):
            with gr.Row(equal_height=False):
                # 左侧列：文件上传与操作区域
                with gr.Column(scale=1, min_width=320):
                    files_in = gr.File(
                        label="待检测音频输入端（点击多选文件，或拖入音频/文件夹）",
                        file_count="multiple",
                        visible=True,
                    )
                    dir_btn = gr.UploadButton(
                        "📂 选择文件夹上传",
                        file_count="directory",
                        variant="secondary",
                        elem_classes="dir-upload-btn",
                    )
                    btn_run = gr.Button("开始智能检测", variant="primary", size="lg", interactive=False)
                
                # 右侧列：实时流式渲染与诊断分析列
                with gr.Column(scale=2.2, min_width=450):
                    result_html = gr.HTML(visible=False)
                    csv_out = gr.File(
                        label="下载结构化 CSV 诊断报表",
                        visible=False,
                    )

        # 当通过选择文件夹按钮上传时，自动把文件同步到上方的 files_in 预览框中显示
        dir_btn.upload(
            fn=lambda x: x,
            inputs=dir_btn,
            outputs=files_in,
        )

        # 绑定点击预测事件，使用生成器函数实现瞬时实时流式渲染，彻底防止二次点击与卡死
        btn_run.click(
            fn=predict_unified,
            inputs=files_in,
            outputs=[btn_run, result_html, csv_out],
        )
        
        # 绑定清空/改变文件事件，实现清空文件时自动隐藏结果面板并置灰禁用按钮
        files_in.change(
            on_files_change,
            inputs=files_in,
            outputs=[result_html, csv_out, btn_run],
        )

    return demo


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None, help="可选：直接检测本地音频文件或文件夹路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument(
        "--route",
        choices=["text", "audio"],
        default="text",
        help="text=Qwen3-ASR+多语言翻译+M1 E1B；audio=V334C纯音频",
    )
    args = parser.parse_args()

    # 启动时立即加载模型
    print("=" * 50)
    print("正在加载模型，请稍候...")
    print(f"  推理路线 : {args.route}")
    if args.route == "text":
        print(f"  ASR 模型 : {ASR_MODEL_PATH}")
        print(f"  翻译模型: {TRANSLATOR_MODEL_PATH}")
        print(f"  文本模型 : {CLS_MODEL_PATH}")
    else:
        print(f"  音频基座 : {AUDIO_BASE_PATH}")
        print(f"  音频参数 : {AUDIO_CHECKPOINT_PATH}")
    print("=" * 50)

    _detector = TeleFraudDetector(
        asr_model_path=ASR_MODEL_PATH,
        translator_model_path=TRANSLATOR_MODEL_PATH,
        cls_model_path=CLS_MODEL_PATH,
        audio_base_path=AUDIO_BASE_PATH,
        audio_checkpoint_path=AUDIO_CHECKPOINT_PATH,
        route=args.route,
        device=args.device,
    )

    print("=" * 50)
    print(f"模型加载完成，设备：{_detector.device}")
    
    if args.path:
        print("=" * 50)
        print(f"正在以命令行模式运行检测：{args.path}")
        path = Path(args.path)
        if not path.exists():
            print(f"错误：路径不存在 {args.path}")
            sys.exit(1)
            
        audio_paths = []
        if path.is_file():
            if path.suffix.lower() in AUDIO_EXTS:
                audio_paths.append(str(path))
        elif path.is_dir():
            audio_paths = sorted(
                str(p) for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )
            
        if not audio_paths:
            print("错误：未找到任何支持的音频文件！")
            sys.exit(1)
            
        print(f"找到 {len(audio_paths)} 条音频，正在检测中...")
        
        # 运行批量预测，无 Gradio progress 进度条，使用流式生成器完成推理
        results = []
        csv_path = None
        for res, path in _run_batch(audio_paths, progress=None):
            results = res
            csv_path = path
        
        print("=" * 50)
        print("检测结果汇总：")
        print("=" * 50)
        for r in results:
            tag = "🚨 涉诈" if r["is_fraud"] else "✅ 安全"
            print(f" 文件名 : {Path(r['audio_path']).name}")
            print(f" 结  果 : {tag} | 分类 : {r['label_name']} | 耗时 : {r['elapsed_s']}s")
            print(f" 转  写 : {r['transcript']}")
            print("-" * 50)
        print(f"检测完成！结果已保存至 CSV 报表：{csv_path}")
        print("=" * 50)
        sys.exit(0)

    print(f"启动 Web 服务，端口：{args.port}")
    print(f"浏览器访问：http://localhost:{args.port}")
    print("=" * 50)

    model_info = f"设备：{_detector.device.upper()}"
    ui = build_ui(model_info)
    # allowed_paths 允许 Gradio 访问项目 outputs/ 和 caches/ 目录
    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(CACHE_DIR).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=[str(out_dir), str(cache_dir)],
    )
