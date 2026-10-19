"""Pure web input/export helpers; no model imports, downloads, or training."""
import csv
from datetime import datetime
import json
from pathlib import Path
import uuid

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_RECORDS = 200
MAX_TEXT_CHARS = 200000
MAX_ID_CHARS = 128
CSV_FIELDS = [
    'sample_id', 'prediction_binary', 'label', 'p_normal', 'p_fraud',
    'threshold', 'raw_wordpiece_length', 'num_chunks', 'coverage',
    'diagnostic_max_chunks_exceeded', 'model_version', 'elapsed_ms',
]
TABLE_HEADERS = ['样本编号', '二分类结果', '诈骗分数', '正常分数',
                 'token 数', '分块数', '覆盖率', '模型推理毫秒/通']


class WebInputError(ValueError):
    """Only fixed safe messages, never original record contents or paths."""


def _checked_record(record, normalize_fn, number):
    try:
        normalized = normalize_fn(record)
    except (ValueError, TypeError, KeyError):
        raise WebInputError('第 {} 条格式不符合要求，请检查样本编号和文本/轮次字段。'.format(number)) from None
    sid = normalized['sample_id']
    if len(sid) > MAX_ID_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in sid):
        raise WebInputError('第 {} 条样本编号须不超过 128 字符，且不能包含控制字符。'.format(number))
    if len(normalized['model_input_text']) > MAX_TEXT_CHARS:
        raise WebInputError('第 {} 条超过每通 200,000 字符上限；已拒绝，不会截断。'.format(number))
    # Only the normalized input survives validation. Supplied labels and metadata
    # are deliberately discarded, so they cannot influence inference or exports.
    return normalized


def single_text_record(text, sample_id=None, normalize_fn=None):
    if not isinstance(text, str) or not text.strip():
        raise WebInputError('请先输入完整通话文本。')
    if len(text) > MAX_TEXT_CHARS:
        raise WebInputError('输入超过每通 200,000 字符上限；已拒绝，不会截断。')
    sid = sample_id or 'web_' + uuid.uuid4().hex
    record = {'sample_id': sid, 'text': text}
    if normalize_fn is None:
        return record
    return _checked_record(record, normalize_fn, 1)


def parse_jsonl_upload(path, normalize_fn, max_bytes=MAX_UPLOAD_BYTES,
                       max_records=MAX_RECORDS, max_chars=MAX_TEXT_CHARS):
    if path is None:
        raise WebInputError('请先上传 UTF-8 JSONL 文件。')
    if max_chars != MAX_TEXT_CHARS:
        raise ValueError('max_chars must equal the frozen web service limit')
    upload = Path(path)
    if upload.suffix.lower() != '.jsonl':
        raise WebInputError('仅支持 .jsonl 文件；音频和其他文件类型不受支持。')
    try:
        if upload.stat().st_size > max_bytes:
            raise WebInputError('上传文件超过 20 MiB 上限；已拒绝，不会截断。')
        # Bound the actual read too, in case a file changes after stat().
        with upload.open('rb') as stream:
            content = stream.read(max_bytes + 1)
    except OSError:
        raise WebInputError('无法读取上传文件，请重新上传。') from None
    if len(content) > max_bytes:
        raise WebInputError('上传文件超过 20 MiB 上限；已拒绝，不会截断。')
    try:
        content = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise WebInputError('文件必须为 UTF-8 编码，可包含 UTF-8 BOM。') from None
    records = []
    seen = set()
    # JSONL physical lines use LF (with optional CR). splitlines() would also
    # split legal Unicode U+2028/U+2029/U+0085 inside JSON string values.
    for line_number, line in enumerate(content.split('\n'), 1):
        if not line.strip():
            continue
        if len(records) >= max_records:
            raise WebInputError('单批超过 200 通上限；请拆分为独立文件。')
        try:
            record = json.loads(line)
        except (ValueError, TypeError, RecursionError):
            raise WebInputError('第 {} 行不是有效的 JSON 对象。'.format(line_number)) from None
        normalized = _checked_record(record, normalize_fn, line_number)
        if normalized['sample_id'] in seen:
            raise WebInputError('第 {} 行样本编号重复；请确保每通编号唯一。'.format(line_number))
        seen.add(normalized['sample_id'])
        records.append(normalized)
    if not records:
        raise WebInputError('文件没有可评测的样本。')
    return records


def _csv_safe(value):
    # Avoid spreadsheet formula injection while preserving the exact ID in JSONL.
    if isinstance(value, str) and value.lstrip(' \t\r\n').startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def export_predictions(predictions, output_root):
    """Write only whitelisted result fields, never input records or transcripts."""
    rows = [{key: pred[key] for key in CSV_FIELDS} for pred in predictions]
    # Validate serialization before creating either export file.
    lines = [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex
    jsonl_path = output_root / ('web_results_' + run_id + '.jsonl')
    csv_path = output_root / ('web_results_' + run_id + '.csv')
    # x mode and server-generated names cannot overwrite another run's results.
    with jsonl_path.open('x', encoding='utf-8', newline='\n') as stream:
        for line in lines:
            stream.write(line + '\n')
    csv_created = False
    try:
        with csv_path.open('x', encoding='utf-8-sig', newline='') as stream:
            csv_created = True
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({k: _csv_safe(v) for k, v in row.items()} for row in rows)
    except Exception:
        # Both paths were created exclusively by this request.
        jsonl_path.unlink(missing_ok=True)
        if csv_created:
            csv_path.unlink(missing_ok=True)
        raise
    return str(jsonl_path), str(csv_path)


def display_rows(predictions):
    return [[pred['sample_id'], '诈骗' if pred['prediction_binary'] else '正常',
             round(pred['p_fraud'], 6), round(pred['p_normal'], 6),
             pred['raw_wordpiece_length'], pred['num_chunks'], pred['coverage'],
             round(pred['elapsed_ms'], 2)] for pred in predictions]
