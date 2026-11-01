"""上传文件的展示名称与内部存储路径分离。"""
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from .document_loader import SUPPORTED_EXTENSIONS, load_and_split_files
from .models import mask_secret


def save_upload(upload, folder: Path) -> Path:
    name = str(upload.name)
    if (not name or name in {".", ".."} or "/" in name or "\\" in name
            or PureWindowsPath(name).drive or any(ord(c) < 32 for c in name)):
        raise ValueError("文件名不允许包含路径、盘符或控制字符。")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{suffix}")
    data = upload.getbuffer()
    if not len(data):
        raise ValueError("上传文件为 0 字节空文件。")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # 排他创建：即使随机名称碰撞，也绝不覆盖已有文件。
    for _ in range(5):
        target = folder / f"{uuid4().hex}{suffix}"
        try:
            with target.open("xb") as handle:
                handle.write(data)
            return target
        except FileExistsError:
            continue
    raise FileExistsError("无法分配唯一的上传文件名，请重试。")


def ingest_uploads(uploads, folder: Path, chunk_size=600, chunk_overlap=100):
    paths, source_names, failures = [], {}, []
    for upload in uploads:
        try:
            path = save_upload(upload, folder)
            paths.append(path)
            source_names[path.name] = str(upload.name)
        except Exception as exc:
            failures.append({"file": str(upload.name), "reason": mask_secret(str(exc))})
    docs, report = load_and_split_files(
        paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        source_names=source_names,
    )
    report["failed"] = failures + report["failed"]
    return docs, report
