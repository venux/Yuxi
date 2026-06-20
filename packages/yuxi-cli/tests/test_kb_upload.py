from __future__ import annotations

import io
import threading
import time
from collections import Counter
from pathlib import Path

from rich.console import Console

from yuxi_cli.config import ConfigStore, Remote
from yuxi_cli.kb_upload import (
    ExtensionOption,
    KbUploadSummary,
    KbUploadOptions,
    LocalFile,
    SkippedFile,
    _format_unsupported_summary,
    _print_selection_summary,
    _render_database_select_lines,
    _render_extension_select_lines,
    run_kb_upload,
)


class FakeKbClient:
    uploaded: list[str] = []
    add_payload: dict | None = None
    active_uploads = 0
    max_active_uploads = 0
    lock = threading.Lock()

    def __init__(self, remote: Remote):
        self.remote = remote

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    @classmethod
    def reset(cls) -> None:
        cls.uploaded = []
        cls.add_payload = None
        cls.active_uploads = 0
        cls.max_active_uploads = 0

    def discovery(self):
        return {
            "version": "0.7.1",
            "capabilities": {"cli": {"kb_upload": True}},
        }

    def get_database(self, kb_id: str):
        return {"kb_id": kb_id, "name": "Test KB", "kb_type": "milvus"}

    def list_databases(self):
        return {
            "databases": [
                {"kb_id": "kb_1", "name": "Milvus KB", "kb_type": "milvus"},
                {"kb_id": "dify_1", "name": "Dify KB", "kb_type": "dify"},
            ]
        }

    def get_knowledge_base_types(self):
        return {
            "kb_types": {
                "milvus": {"supports_documents": True},
                "dify": {"supports_documents": False},
            }
        }

    def get_supported_file_types(self):
        return {
            "file_types": [
                ".bmp",
                ".csv",
                ".docx",
                ".html",
                ".htm",
                ".jpeg",
                ".jpg",
                ".json",
                ".md",
                ".pdf",
                ".png",
                ".pptx",
                ".tif",
                ".tiff",
                ".txt",
                ".xls",
                ".xlsx",
            ]
        }

    def upload_knowledge_file(self, kb_id: str, path: Path):
        with self.lock:
            type(self).active_uploads += 1
            type(self).max_active_uploads = max(type(self).max_active_uploads, type(self).active_uploads)
        try:
            time.sleep(0.02)
            type(self).uploaded.append(path.name)
            return {
                "file_path": f"minio://knowledgebases/{kb_id}/upload/{path.name}",
                "content_hash": f"hash-{path.name}",
                "size": path.stat().st_size,
            }
        finally:
            with self.lock:
                type(self).active_uploads -= 1

    def add_uploaded_documents(self, kb_id: str, items: list[str], params: dict):
        type(self).add_payload = {"kb_id": kb_id, "items": items, "params": params}
        return {"status": "success", "added": len(items), "failed": 0, "items": [], "failed_items": []}


def _console():
    return Console(file=io.StringIO(), force_terminal=False)


def _store(tmp_path: Path) -> ConfigStore:
    store = ConfigStore(tmp_path / "config.toml")
    config = store.load()
    remote = config.get_remote("local")
    remote.api_key = "yxkey_test"
    store.save(config)
    return store


def test_kb_upload_default_include_excludes_structured_and_presentation_files(tmp_path):
    FakeKbClient.reset()
    for name in [
        "a.md",
        "b.txt",
        "c.docx",
        "d.html",
        "e.htm",
        "f.json",
        "g.csv",
        "h.xls",
        "i.xlsx",
        "j.pptx",
        "k.pdf",
    ]:
        (tmp_path / name).write_text("demo", encoding="utf-8")

    run_kb_upload(
        _store(tmp_path),
        None,
        KbUploadOptions(path=tmp_path, kb_id="kb_1", yes=True, concurrency=2),
        _console(),
        client_factory=FakeKbClient,
    )

    assert sorted(FakeKbClient.uploaded) == ["a.md", "b.txt", "c.docx", "d.html", "e.htm"]
    assert FakeKbClient.add_payload is not None
    assert len(FakeKbClient.add_payload["items"]) == 5


def test_kb_upload_preserves_relative_source_paths(tmp_path):
    FakeKbClient.reset()
    docs_dir = tmp_path / "docs" / "guide"
    docs_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("intro", encoding="utf-8")
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")

    run_kb_upload(
        _store(tmp_path),
        None,
        KbUploadOptions(path=tmp_path, kb_id="kb_1", yes=True, concurrency=2),
        _console(),
        client_factory=FakeKbClient,
    )

    assert FakeKbClient.add_payload is not None
    source_paths = FakeKbClient.add_payload["params"]["source_paths"]
    assert source_paths == {
        "minio://knowledgebases/kb_1/upload/intro.md": "docs/guide/intro.md",
        "minio://knowledgebases/kb_1/upload/root.txt": "root.txt",
    }


def test_kb_upload_without_kb_id_selects_only_uploadable_database(tmp_path):
    FakeKbClient.reset()
    (tmp_path / "note.md").write_text("demo", encoding="utf-8")

    run_kb_upload(
        _store(tmp_path),
        None,
        KbUploadOptions(path=tmp_path, yes=True, concurrency=2),
        _console(),
        client_factory=FakeKbClient,
    )

    assert FakeKbClient.add_payload is not None
    assert FakeKbClient.add_payload["kb_id"] == "kb_1"
    assert FakeKbClient.add_payload["items"] == ["minio://knowledgebases/kb_1/upload/note.md"]


def test_kb_upload_include_ext_allows_non_default_supported_types(tmp_path):
    FakeKbClient.reset()
    (tmp_path / "data.xlsx").write_text("demo", encoding="utf-8")
    (tmp_path / "slides.pptx").write_text("demo", encoding="utf-8")
    (tmp_path / "note.md").write_text("demo", encoding="utf-8")

    run_kb_upload(
        _store(tmp_path),
        None,
        KbUploadOptions(path=tmp_path, kb_id="kb_1", yes=True, concurrency=2, include_ext="xlsx,pptx"),
        _console(),
        client_factory=FakeKbClient,
    )

    assert sorted(FakeKbClient.uploaded) == ["data.xlsx", "slides.pptx"]


def test_kb_upload_limits_upload_concurrency(tmp_path):
    FakeKbClient.reset()
    for index in range(6):
        (tmp_path / f"{index}.md").write_text("demo", encoding="utf-8")

    run_kb_upload(
        _store(tmp_path),
        None,
        KbUploadOptions(path=tmp_path, kb_id="kb_1", yes=True, concurrency=2),
        _console(),
        client_factory=FakeKbClient,
    )

    assert FakeKbClient.max_active_uploads <= 2
    assert FakeKbClient.max_active_uploads > 1


def test_database_select_lines_use_arrow_selection_without_numbering():
    lines = _render_database_select_lines(
        [
            {"kb_id": "kb_1", "name": "Alpha", "kb_type": "milvus"},
            {"kb_id": "kb_2", "name": "Beta", "kb_type": "milvus"},
        ],
        selected_index=1,
    )

    assert lines[0] == "选择知识库"
    assert "↑/↓" in lines[1]
    assert "1" not in lines[3].split("Alpha", 1)[0]
    assert lines[4].startswith("\x1b[7m› Beta  [milvus]  kb_2")


def test_extension_select_lines_show_checkbox_counts_and_unsupported_summary():
    lines = _render_extension_select_lines(
        [
            ExtensionOption(".html", 101),
            ExtensionOption(".md", 165),
            ExtensionOption(".txt", 2),
            ExtensionOption(".json", 9),
        ],
        selected_extensions={".html", ".md", ".txt"},
        cursor=0,
        unsupported_counts=Counter({".py": 3831}),
    )

    assert "Space 选择/取消" in lines[1]
    assert "[x] html (101)" in lines[3]
    assert "[x] md (165)" in lines[4]
    assert "[x] txt (2)" in lines[5]
    assert "[ ] json (9)" in lines[6]
    assert lines[-1] == "- 不支持 3831 (.py)"


def test_unsupported_summary_truncates_extensions_without_per_extension_counts():
    summary = _format_unsupported_summary(
        Counter(
            {
                ".py": 100,
                ".json": 90,
                ".js": 80,
                ".ts": 70,
                ".map": 60,
                ".css": 50,
                ".yaml": 40,
                ".lock": 30,
                ".mjs": 20,
                ".cjs": 10,
            }
        )
    )

    assert summary == "- 不支持 550 (.py, .json, .js, .ts, .map, .css, .yaml, .lock, 等 2 类)"
    assert ".py 100" not in summary
    assert ".json 90" not in summary


def test_selection_summary_lists_selected_and_unselected_supported_types(tmp_path):
    selected = [LocalFile(tmp_path / "a.md", "a.md", ".md", 4)]
    skipped = [
        SkippedFile(tmp_path / "b.json", "b.json", "not-included"),
        SkippedFile(tmp_path / "c.py", "c.py", "unsupported"),
    ]
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False)

    _print_selection_summary(KbUploadSummary(scanned=3, selected=selected, skipped=skipped), console)

    output = buffer.getvalue()
    assert "文件类型:" in output
    assert "  [x] md (1)" in output
    assert "  [ ] json (1)" in output
    assert "- 不支持 1 (.py)" in output
