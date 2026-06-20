from __future__ import annotations

import curses
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console

from yuxi_cli.client import ClientError, YuxiClient
from yuxi_cli.config import ConfigStore, Remote
from yuxi_cli.discovery import ServerCompatibilityError, ensure_server_compatible

DEFAULT_INCLUDE_EXTENSIONS = {".docx", ".html", ".htm", ".md", ".txt"}
DEFAULT_EXCLUDE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff"}
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
MAX_CONCURRENCY = 10
UPLOAD_RETRY_ATTEMPTS = 2


class KbUploadError(Exception):
    pass


@dataclass(frozen=True)
class KbUploadOptions:
    path: Path
    kb_id: str | None = None
    yes: bool = False
    concurrency: int = 3
    include_ext: str | None = None
    exclude_ext: str | None = None


@dataclass(frozen=True)
class LocalFile:
    path: Path
    relative_path: str
    extension: str
    size: int


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    relative_path: str
    reason: str


@dataclass(frozen=True)
class ExtensionOption:
    extension: str
    count: int


@dataclass
class UploadResult:
    local_file: LocalFile
    file_path: str | None = None
    content_hash: str | None = None
    size: int | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.file_path and self.content_hash)


@dataclass
class KbUploadSummary:
    scanned: int
    skipped: list[SkippedFile] = field(default_factory=list)
    selected: list[LocalFile] = field(default_factory=list)
    uploaded: list[UploadResult] = field(default_factory=list)
    upload_failed: list[UploadResult] = field(default_factory=list)
    add_response: dict | None = None

    @property
    def add_failed_count(self) -> int:
        if not self.add_response:
            return 0
        return int(self.add_response.get("failed") or 0)


def run_kb_upload(
    store: ConfigStore,
    remote_name: str | None,
    options: KbUploadOptions,
    console: Console,
    *,
    client_factory=YuxiClient,
) -> KbUploadSummary:
    if options.concurrency < 1 or options.concurrency > MAX_CONCURRENCY:
        raise KbUploadError(f"--concurrency 必须在 1 到 {MAX_CONCURRENCY} 之间")

    config = store.load()
    remote = config.get_remote(remote_name)
    if not remote.api_key:
        raise KbUploadError(f"remote 尚未登录: {remote.name}")

    with client_factory(remote) as client:
        _ensure_kb_upload_supported(client)
        kb_types = _load_kb_types(client)
        database = _resolve_database(client, options.kb_id, kb_types, console)
        kb_id = str(database.get("kb_id") or "")
        if not kb_id:
            raise KbUploadError("知识库详情缺少 kb_id")
        supported_extensions = _load_supported_extensions(client)

    scanned_files, initial_skipped = scan_local_files(options.path)
    selected_extensions = None
    if not options.yes:
        selected_extensions = _prompt_select_extensions(
            scanned_files,
            initial_skipped,
            supported_extensions=supported_extensions,
            include_ext=options.include_ext,
            exclude_ext=options.exclude_ext,
        )
    selected, skipped_by_type = select_upload_files(
        scanned_files,
        supported_extensions=supported_extensions,
        include_ext=options.include_ext,
        exclude_ext=options.exclude_ext,
        selected_extensions=selected_extensions,
    )
    summary = KbUploadSummary(
        scanned=len(scanned_files) + len(initial_skipped),
        skipped=[*initial_skipped, *skipped_by_type],
        selected=selected,
    )

    if not selected:
        _print_selection_summary(summary, console)
        raise KbUploadError("没有可上传的文件")

    _print_selection_summary(summary, console)
    if not options.yes and not typer.confirm("继续上传?", default=True):
        raise KbUploadError("已取消")

    uploaded, failed = upload_files(
        remote,
        client_factory,
        kb_id,
        selected,
        concurrency=options.concurrency,
        console=console,
    )
    summary.uploaded = uploaded
    summary.upload_failed = failed

    if not uploaded:
        _print_final_summary(summary, console)
        raise KbUploadError("所有文件上传失败，未添加文档记录")

    with client_factory(remote) as client:
        summary.add_response = add_uploaded_documents(client, kb_id, uploaded)

    _print_final_summary(summary, console)
    if failed or summary.add_failed_count:
        raise KbUploadError("部分文件处理失败，请查看摘要")
    return summary


def scan_local_files(path: Path) -> tuple[list[LocalFile], list[SkippedFile]]:
    root = path.expanduser()
    if not root.exists():
        raise KbUploadError(f"路径不存在: {path}")

    if root.is_file():
        return _local_file_from_path(root, root.name)

    if not root.is_dir():
        raise KbUploadError(f"路径不是文件或目录: {path}")

    files: list[LocalFile] = []
    skipped: list[SkippedFile] = []
    for current in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative_path = current.relative_to(root).as_posix()
        if _has_hidden_part(relative_path):
            skipped.append(SkippedFile(current, relative_path, "hidden"))
            continue
        item_files, item_skipped = _local_file_from_path(current, relative_path)
        files.extend(item_files)
        skipped.extend(item_skipped)
    return files, skipped


def _has_hidden_part(relative_path: str) -> bool:
    return any(part.startswith(".") for part in Path(relative_path).parts)


def select_upload_files(
    files: list[LocalFile],
    *,
    supported_extensions: set[str],
    include_ext: str | None,
    exclude_ext: str | None,
    selected_extensions: set[str] | None = None,
) -> tuple[list[LocalFile], list[SkippedFile]]:
    if selected_extensions is None:
        include_extensions = parse_extension_list(include_ext) if include_ext else set(DEFAULT_INCLUDE_EXTENSIONS)
        exclude_extensions = parse_extension_list(exclude_ext) if exclude_ext else (
            set() if include_ext else set(DEFAULT_EXCLUDE_EXTENSIONS)
        )
    else:
        include_extensions = selected_extensions
        exclude_extensions = set()

    selected: list[LocalFile] = []
    skipped: list[SkippedFile] = []
    for item in files:
        if item.extension not in supported_extensions:
            skipped.append(SkippedFile(item.path, item.relative_path, "unsupported"))
            continue
        if item.extension not in include_extensions:
            reason = "not-selected" if selected_extensions is not None else "not-included"
            skipped.append(SkippedFile(item.path, item.relative_path, reason))
            continue
        if item.extension in exclude_extensions:
            skipped.append(SkippedFile(item.path, item.relative_path, "excluded"))
            continue
        selected.append(item)
    return selected, skipped


def parse_extension_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    extensions = set()
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        extensions.add(value)
    return extensions


def _prompt_select_extensions(
    files: list[LocalFile],
    initial_skipped: list[SkippedFile],
    *,
    supported_extensions: set[str],
    include_ext: str | None,
    exclude_ext: str | None,
) -> set[str]:
    options = _extension_options(files, supported_extensions)
    if not options:
        return set()
    if not sys.stdin.isatty():
        raise KbUploadError("非交互环境请传 --yes 或 --include-ext")

    selected_extensions = _initial_selected_extensions(
        {option.extension for option in options},
        include_ext=include_ext,
        exclude_ext=exclude_ext,
    )
    unsupported_counts = _unsupported_extension_counts(files, initial_skipped, supported_extensions)
    try:
        selected = curses.wrapper(
            _run_extension_selector,
            options,
            selected_extensions,
            unsupported_counts,
        )
    except curses.error as exc:
        raise KbUploadError("当前终端不支持文件类型多选，请传 --yes 或 --include-ext") from exc
    if selected is None:
        raise KbUploadError("已取消")
    return selected


def _extension_options(files: list[LocalFile], supported_extensions: set[str]) -> list[ExtensionOption]:
    counts = Counter(item.extension for item in files if item.extension in supported_extensions)
    return [ExtensionOption(extension, count) for extension, count in sorted(counts.items())]


def _initial_selected_extensions(
    available_extensions: set[str], *, include_ext: str | None, exclude_ext: str | None
) -> set[str]:
    selected = parse_extension_list(include_ext) if include_ext else set(DEFAULT_INCLUDE_EXTENSIONS)
    if exclude_ext:
        selected -= parse_extension_list(exclude_ext)
    return selected & available_extensions


def _unsupported_extension_counts(
    files: list[LocalFile], initial_skipped: list[SkippedFile], supported_extensions: set[str]
) -> Counter[str]:
    counts = Counter(item.extension for item in files if item.extension not in supported_extensions)
    no_extension_count = sum(1 for item in initial_skipped if item.reason == "no-extension")
    if no_extension_count:
        counts["无扩展名"] = no_extension_count
    return counts


def upload_files(
    remote: Remote,
    client_factory,
    kb_id: str,
    files: list[LocalFile],
    *,
    concurrency: int,
    console: Console,
) -> tuple[list[UploadResult], list[UploadResult]]:
    uploaded: list[UploadResult] = []
    failed: list[UploadResult] = []
    completed = 0
    progress_step = max(1, len(files) // 10)

    def upload_one(item: LocalFile) -> UploadResult:
        return _upload_one_with_retry(remote, client_factory, kb_id, item)

    console.print(f"开始上传: {len(files)} 个文件，并发 {concurrency}")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {executor.submit(upload_one, item): item for item in files}
            for future in as_completed(future_map):
                completed += 1
                result = future.result()
                if result.success:
                    uploaded.append(result)
                else:
                    failed.append(result)
                    console.print(
                        f"[red]✗[/red] {result.local_file.relative_path} ({completed}/{len(files)}): {result.error}"
                    )
                if completed == len(files) or completed % progress_step == 0:
                    console.print(f"上传进度: {completed}/{len(files)}")
    except KeyboardInterrupt as exc:
        raise KbUploadError("已取消上传队列，未添加文档记录") from exc

    uploaded.sort(key=lambda item: item.local_file.relative_path)
    failed.sort(key=lambda item: item.local_file.relative_path)
    return uploaded, failed


def add_uploaded_documents(client: YuxiClient, kb_id: str, uploaded: list[UploadResult]) -> dict:
    items = [result.file_path for result in uploaded if result.file_path]
    params = {
        "content_type": "file",
        "content_hashes": {result.file_path: result.content_hash for result in uploaded if result.file_path},
        "file_sizes": {result.file_path: result.size for result in uploaded if result.file_path},
        "source_paths": {result.file_path: result.local_file.relative_path for result in uploaded if result.file_path},
    }
    return client.add_uploaded_documents(kb_id, items, params)


def _local_file_from_path(path: Path, relative_path: str) -> tuple[list[LocalFile], list[SkippedFile]]:
    if path.is_symlink():
        return [], [SkippedFile(path, relative_path, "symlink")]
    try:
        stat = path.stat()
    except OSError as exc:
        return [], [SkippedFile(path, relative_path, f"stat-failed: {exc}")]
    if stat.st_size == 0:
        return [], [SkippedFile(path, relative_path, "empty")]
    if stat.st_size > MAX_UPLOAD_SIZE_BYTES:
        return [], [SkippedFile(path, relative_path, "too-large")]
    if not os.access(path, os.R_OK):
        return [], [SkippedFile(path, relative_path, "unreadable")]
    extension = path.suffix.lower()
    if not extension:
        return [], [SkippedFile(path, relative_path, "no-extension")]
    return [LocalFile(path=path, relative_path=relative_path.replace("\\", "/"), extension=extension, size=stat.st_size)], []


def _ensure_kb_upload_supported(client: YuxiClient) -> None:
    try:
        ensure_server_compatible(client.discovery(), "cli.kb_upload")
    except ServerCompatibilityError as exc:
        raise KbUploadError(str(exc)) from exc


def _load_kb_types(client: YuxiClient) -> dict:
    payload = client.get_knowledge_base_types()
    kb_types = payload.get("kb_types")
    if not isinstance(kb_types, dict):
        raise KbUploadError("服务端知识库类型响应格式无效")
    return kb_types


def _resolve_database(client: YuxiClient, kb_id: str | None, kb_types: dict, console: Console) -> dict:
    if kb_id and kb_id.strip():
        database = client.get_database(kb_id.strip())
        _ensure_database_supports_documents(database, kb_types)
        return database

    databases = _list_uploadable_databases(client, kb_types)
    if not databases:
        raise KbUploadError("当前 remote 没有可用于文档上传的知识库")
    if len(databases) == 1:
        database = databases[0]
        console.print(f"已选择唯一可用知识库: {database.get('name') or database.get('kb_id')}")
        return database
    if not sys.stdin.isatty():
        raise KbUploadError("非交互环境必须显式传入 --kb-id")

    return _prompt_select_database(databases)


def _prompt_select_database(databases: list[dict]) -> dict:
    try:
        selected_index = curses.wrapper(_run_database_selector, databases)
    except curses.error as exc:
        raise KbUploadError("当前终端不支持方向键单选，请显式传入 --kb-id") from exc
    if selected_index is None:
        raise KbUploadError("已取消")
    return databases[selected_index]


def _run_database_selector(stdscr, databases: list[dict]) -> int | None:
    selected_index = 0
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        _draw_database_selector(stdscr, databases, selected_index, height, width)
        key = stdscr.getch()
        if key in {curses.KEY_UP, ord("k")}:
            selected_index = (selected_index - 1) % len(databases)
        elif key in {curses.KEY_DOWN, ord("j")}:
            selected_index = (selected_index + 1) % len(databases)
        elif key in {curses.KEY_ENTER, 10, 13}:
            return selected_index
        elif key in {27, ord("q"), ord("Q"), 3}:
            return None


def _draw_database_selector(stdscr, databases: list[dict], selected_index: int, height: int, width: int) -> None:
    lines = _render_database_select_lines(databases, selected_index)
    max_lines = max(0, height - 1)
    max_width = max(1, width - 1)
    for row, line in enumerate(lines[:max_lines]):
        clean_line = _strip_ansi(line)
        attr = curses.A_REVERSE if row >= 3 and row - 3 == selected_index else curses.A_NORMAL
        stdscr.addnstr(row, 0, clean_line, max_width, attr)
    stdscr.refresh()


def _render_database_select_lines(databases: list[dict], selected_index: int) -> list[str]:
    lines = [
        "选择知识库",
        "↑/↓ 移动 · Enter 确认 · Esc/q 取消",
        "",
    ]
    for index, database in enumerate(databases):
        marker = "›" if index == selected_index else " "
        label = _database_option_label(database)
        if index == selected_index:
            lines.append(f"\x1b[7m{marker} {label}\x1b[0m")
        else:
            lines.append(f"{marker} {label}")
    return lines


def _database_option_label(database: dict) -> str:
    name = str(database.get("name") or database.get("database_name") or "-")
    kb_id = str(database.get("kb_id") or "-")
    kb_type = str(database.get("kb_type") or "-")
    return f"{name}  [{kb_type}]  {kb_id}"


def _run_extension_selector(
    stdscr,
    options: list[ExtensionOption],
    selected_extensions: set[str],
    unsupported_counts: Counter[str],
) -> set[str] | None:
    cursor = 0
    selected = set(selected_extensions)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        _draw_extension_selector(stdscr, options, selected, cursor, unsupported_counts, height, width)
        key = stdscr.getch()
        if key in {curses.KEY_UP, ord("k")}:
            cursor = (cursor - 1) % len(options)
        elif key in {curses.KEY_DOWN, ord("j")}:
            cursor = (cursor + 1) % len(options)
        elif key in {ord(" ")}:
            extension = options[cursor].extension
            if extension in selected:
                selected.remove(extension)
            else:
                selected.add(extension)
        elif key in {curses.KEY_ENTER, 10, 13}:
            return selected
        elif key in {27, ord("q"), ord("Q"), 3}:
            return None


def _draw_extension_selector(
    stdscr,
    options: list[ExtensionOption],
    selected_extensions: set[str],
    cursor: int,
    unsupported_counts: Counter[str],
    height: int,
    width: int,
) -> None:
    lines = _render_extension_select_lines(options, selected_extensions, cursor, unsupported_counts)
    max_lines = max(0, height - 1)
    max_width = max(1, width - 1)
    for row, line in enumerate(lines[:max_lines]):
        clean_line = _strip_ansi(line)
        attr = curses.A_REVERSE if row >= 3 and row - 3 == cursor else curses.A_NORMAL
        stdscr.addnstr(row, 0, clean_line, max_width, attr)
    stdscr.refresh()


def _render_extension_select_lines(
    options: list[ExtensionOption],
    selected_extensions: set[str],
    cursor: int,
    unsupported_counts: Counter[str],
) -> list[str]:
    lines = [
        "选择要上传的文件类型",
        "↑/↓ 移动 · Space 选择/取消 · Enter 确认 · Esc/q 取消",
        "",
    ]
    for index, option in enumerate(options):
        marker = "›" if index == cursor else " "
        checked = "x" if option.extension in selected_extensions else " "
        label = option.extension.lstrip(".")
        line = f"{marker} [{checked}] {label} ({option.count})"
        if index == cursor:
            lines.append(f"\x1b[7m{line}\x1b[0m")
        else:
            lines.append(line)
    if unsupported_counts:
        lines.append("")
        lines.append(_format_unsupported_summary(unsupported_counts))
    return lines


def _format_unsupported_summary(unsupported_counts: Counter[str]) -> str:
    total = sum(unsupported_counts.values())
    extensions = [extension for extension, _count in unsupported_counts.most_common()]
    visible = extensions[:8]
    remaining = len(extensions) - len(visible)
    suffix = f", 等 {remaining} 类" if remaining else ""
    return f"- 不支持 {total} ({', '.join(visible)}{suffix})"


def _strip_ansi(value: str) -> str:
    return value.replace("\x1b[7m", "").replace("\x1b[0m", "")


def _list_uploadable_databases(client: YuxiClient, kb_types: dict) -> list[dict]:
    payload = client.list_databases()
    databases = payload.get("databases")
    if not isinstance(databases, list):
        raise KbUploadError("服务端知识库列表响应格式无效")
    uploadable = []
    for database in databases:
        if not isinstance(database, dict):
            continue
        if _database_supports_documents(database, kb_types):
            uploadable.append(database)
    uploadable.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("kb_id") or "")))
    return uploadable


def _ensure_database_supports_documents(database: dict, kb_types: dict) -> None:
    kb_type = str(database.get("kb_type") or "")
    if not kb_type:
        raise KbUploadError("知识库详情缺少 kb_type，无法确认是否支持文档上传")

    type_info = kb_types.get(kb_type)
    if not isinstance(type_info, dict):
        raise KbUploadError(f"服务端未返回知识库类型信息: {kb_type}")
    if type_info.get("supports_documents") is False:
        raise KbUploadError(f"{database.get('name') or kb_type} 只支持检索，不支持文档上传")


def _database_supports_documents(database: dict, kb_types: dict) -> bool:
    kb_type = str(database.get("kb_type") or "")
    type_info = kb_types.get(kb_type)
    return isinstance(type_info, dict) and type_info.get("supports_documents") is True


def _load_supported_extensions(client: YuxiClient) -> set[str]:
    payload = client.get_supported_file_types()
    raw_file_types = payload.get("file_types")
    if not isinstance(raw_file_types, list) or not raw_file_types:
        raise KbUploadError("服务端支持文件类型响应格式无效")
    return {str(item).lower() if str(item).startswith(".") else f".{str(item).lower()}" for item in raw_file_types}


def _upload_one_with_retry(remote: Remote, client_factory, kb_id: str, item: LocalFile) -> UploadResult:
    last_error: Exception | None = None
    for attempt in range(UPLOAD_RETRY_ATTEMPTS + 1):
        try:
            with client_factory(remote) as client:
                data = client.upload_knowledge_file(kb_id, item.path)
            file_path = str(data.get("file_path") or data.get("minio_path") or "")
            content_hash = str(data.get("content_hash") or "")
            size = int(data.get("size") or item.size)
            if not file_path or not content_hash:
                return UploadResult(item, error="上传响应缺少 file_path 或 content_hash")
            return UploadResult(item, file_path=file_path, content_hash=content_hash, size=size)
        except ClientError as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt >= UPLOAD_RETRY_ATTEMPTS:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break
        time.sleep(2**attempt)
    return UploadResult(item, error=str(last_error) if last_error else "上传失败")


def _is_retryable(exc: ClientError) -> bool:
    return exc.status_code is None or exc.status_code == 429 or exc.status_code >= 500


def _print_selection_summary(summary: KbUploadSummary, console: Console) -> None:
    type_options = _upload_type_options_from_summary(summary)
    not_selected = sum(1 for item in summary.skipped if item.reason in {"not-selected", "not-included", "excluded"})
    unsupported_counts = _unsupported_counts_from_skipped(summary.skipped)
    unsupported_total = sum(unsupported_counts.values())
    other_skipped = len(summary.skipped) - not_selected - unsupported_total

    console.print("上传预览")
    console.print(f"- 扫描文件: {summary.scanned}")
    console.print(f"- 将上传: {len(summary.selected)}")
    if not_selected:
        console.print(f"- 未选择: {not_selected}")
    if unsupported_counts:
        console.print(_format_unsupported_summary(unsupported_counts), markup=False)
    if other_skipped:
        console.print(f"- 其他跳过: {other_skipped}")

    if type_options:
        console.print("文件类型:")
        for extension, count, selected in type_options:
            checked = "x" if selected else " "
            console.print(f"  [{checked}] {extension.lstrip('.')} ({count})", markup=False)


def _upload_type_options_from_summary(summary: KbUploadSummary) -> list[tuple[str, int, bool]]:
    counts = Counter(item.extension for item in summary.selected)
    selected_extensions = set(counts)
    for item in summary.skipped:
        if item.reason in {"not-selected", "not-included", "excluded"}:
            extension = item.path.suffix.lower()
            if extension:
                counts[extension] += 1
    return [(extension, counts[extension], extension in selected_extensions) for extension in sorted(counts)]


def _unsupported_counts_from_skipped(skipped: list[SkippedFile]) -> Counter[str]:
    counts = Counter()
    for item in skipped:
        if item.reason == "unsupported":
            counts[item.path.suffix.lower() or "无扩展名"] += 1
        elif item.reason == "no-extension":
            counts["无扩展名"] += 1
    return counts


def _print_final_summary(summary: KbUploadSummary, console: Console) -> None:
    added = int((summary.add_response or {}).get("added") or 0)
    add_failed = int((summary.add_response or {}).get("failed") or 0)

    console.print("上传结果")
    console.print(f"- 上传成功: {len(summary.uploaded)}")
    console.print(f"- 上传失败: {len(summary.upload_failed)}")
    console.print(f"- 添加成功: {added}")
    console.print(f"- 添加失败: {add_failed}")

    failed_items = (summary.add_response or {}).get("failed_items") or []
    if failed_items:
        for item in failed_items:
            console.print(f"[red]添加失败:[/red] {item.get('item')} - {item.get('error')}")
