"""Load a project (uploaded ZIP or explicitly selected local folder) into a bounded index.

Safety rules applied here:
- ZIPs: size, member-count, total-uncompressed and per-member compression-ratio limits;
  absolute paths, drive letters, `..`, symlinks and encrypted members are refused; only files
  that `files.classify` allows are extracted, each written under the workspace folder after
  a resolved-path check.
- Local folders: never the filesystem root or a home folder itself; symlinks are skipped and
  every resolved path must stay inside the root; excluded folders are not even walked.
- Text is decoded, secret values are redacted, and only the redacted text is indexed.
Nothing is executed or imported: Python files are analysed with `ast.parse` only.
"""

from __future__ import annotations

import ast
import hashlib
import io
import math
import os
import re
import shutil
import stat
import time
import tomllib
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from omniflow.code import files

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STOP = frozenset("""the a an and or of to in on for is are be it this that with how what
where which who does do why from by as at file code function class method project explain
show me my about please""".split())


# Exception names in a query -> code shapes that typically raise them (Python files).
EXCEPTION_HINTS = {
    "zerodivisionerror": re.compile(r"[\w)\]]\s*(?<!/)/(?!/)\s*[\w(\[]|\s%\s|divmod\("),
    "keyerror": re.compile(r"\w\[[\"'\w]|\.pop\("),
    "indexerror": re.compile(r"\w\[-?\d+\]|\.pop\("),
    "attributeerror": re.compile(r"\w\.\w+\("),
    "filenotfounderror": re.compile(r"\bopen\(|Path\(|\.read_text\(|\.read_bytes\("),
    "modulenotfounderror": re.compile(r"^\s*(?:from|import)\s"),
    "importerror": re.compile(r"^\s*(?:from|import)\s"),
    "valueerror": re.compile(r"\b(?:int|float|Decimal|datetime\.strptime)\("),
    "jsondecodeerror": re.compile(r"json\.loads?\("),
    "unicodedecodeerror": re.compile(r"\.decode\(|\bopen\("),
    "recursionerror": re.compile(r"^\s*def\s"),
}


class WorkspaceError(ValueError):
    """The project can't be loaded or the request is outside it; message is safe to show."""


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int
    max_files: int
    max_total_chars: int
    zip_max_bytes: int
    zip_max_uncompressed: int
    zip_max_members: int
    read_max_lines: int

    @classmethod
    def from_settings(cls, s) -> Limits:
        return cls(
            max_file_bytes=s.code_max_file_kb * 1024, max_files=s.code_max_files,
            max_total_chars=int(s.code_max_total_mb * 1_000_000),
            zip_max_bytes=int(s.code_zip_max_mb * 1_000_000),
            zip_max_uncompressed=int(s.code_zip_max_uncompressed_mb * 1_000_000),
            zip_max_members=s.code_zip_max_members, read_max_lines=s.code_read_max_lines)


@dataclass
class Symbol:
    name: str  # qualified, e.g. Class.method
    kind: str  # function | class | method
    path: str
    start: int
    end: int
    calls: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class FileEntry:
    path: str
    language: str
    text: str  # redacted
    lines: list[str]
    sha256: str  # of the raw bytes on disk (used to detect changes before patching)
    redactions: int
    syntax_error: str | None = None


@dataclass
class Workspace:
    id: str
    owner: str
    kind: str  # "upload" | "local"
    name: str
    root: Path
    trusted: bool  # tests may run only in trusted local folders
    files: dict[str, FileEntry] = field(default_factory=dict)
    skipped: Counter = field(default_factory=Counter)
    symbols: list[Symbol] = field(default_factory=list)
    imports: dict[str, list[str]] = field(default_factory=dict)
    entry_points: list[dict] = field(default_factory=list)
    truncated: bool = False
    created: float = field(default_factory=time.time)
    _df: Counter = field(default_factory=Counter)
    _tokens: dict[str, Counter] = field(default_factory=dict)
    _defs: dict[str, set] = field(default_factory=dict)  # names defined per file

    # ------------------------------------------------------------- summary
    def languages(self) -> dict[str, int]:
        return dict(Counter(f.language for f in self.files.values()).most_common())

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "files_indexed": len(self.files),
            "languages": self.languages(), "skipped": dict(self.skipped),
            "entry_points": self.entry_points[:15], "truncated": self.truncated,
            "redacted_values": sum(f.redactions for f in self.files.values()),
            "python_syntax_errors": [
                {"path": f.path, "error": f.syntax_error}
                for f in self.files.values() if f.syntax_error][:10],
        }

    # -------------------------------------------------------------- lookups
    def entry(self, path: str) -> FileEntry:
        cleaned = path.replace("\\", "/").strip().lstrip("./")
        found = self.files.get(cleaned)
        if found is None:
            raise WorkspaceError(f"'{path}' is not an indexed file in this project (it may "
                                 "not exist, be excluded, or be unsupported).")
        return found

    def read(self, path: str, start: int = 1, end: int | None = None,
             max_lines: int = 200) -> dict:
        entry = self.entry(path)
        total = len(entry.lines)
        start = max(1, int(start))
        end = min(total, int(end) if end else start + max_lines - 1)
        if start > total and total:
            raise WorkspaceError(f"{entry.path} has only {total} lines.")
        end = min(end, start + max_lines - 1)
        numbered = "\n".join(f"{n:>5}: {entry.lines[n - 1]}" for n in range(start, end + 1))
        return {"path": entry.path, "start_line": start, "end_line": end,
                "total_lines": total, "text": numbered,
                "truncated": end < total and (end - start + 1) >= max_lines}

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        terms = query_terms(query)
        literal = query.strip().strip("\"'`")
        if not terms and not literal:
            return []
        n_files = max(1, len(self.files))
        idf = {t: math.log(1 + n_files / (1 + self._df.get(t, 0))) for t in terms}
        hints = [EXCEPTION_HINTS[t] for t in terms if t in EXCEPTION_HINTS]
        hits = []
        for entry in self.files.values():
            counts = self._tokens.get(entry.path, Counter())
            path_terms = set(split_identifier(entry.path))
            base = sum(idf[t] * (1 + math.log(counts[t])) for t in terms if counts.get(t))
            base += sum(2 * idf[t] for t in terms if t in path_terms)
            defined = self._defs.get(entry.path, set())
            base += sum(3 * idf[t] for t in terms if t in defined)  # defines the name
            code_lines = hints and entry.language == "Python" and not entry.path.startswith(
                ("tests/", "test_"))
            hinted = [n for n, line in enumerate(entry.lines, start=1)
                      if code_lines and not line.lstrip().startswith("#")
                      and any(h.search(line) for h in hints)]
            if hinted:
                base += 3.0  # an error name points at code that can raise it
            if base <= 0 and literal.lower() not in entry.text.lower():
                continue
            best_line, best_score = 1, -1.0
            for number, line in enumerate(entry.lines, start=1):
                low = line.lower()
                words = set(split_identifier(line))
                score = sum(idf[t] for t in terms if t in words)
                if number in hinted:
                    score += 2.0
                if len(literal) > 3 and literal.lower() in low:
                    score += 5
                if re.match(r"\s*(?:async\s+)?(?:def|class)\s", line):
                    score *= 1.5
                if score > best_score:
                    best_line, best_score = number, score
            if len(literal) > 3 and literal.lower() in entry.text.lower():
                base += 5
            hits.append((base + max(0.0, best_score), entry, best_line))
        hits.sort(key=lambda h: (-h[0], h[1].path))
        results = []
        for score, entry, line in hits[:max_results]:
            lo, hi = max(1, line - 3), min(len(entry.lines), line + 6)
            results.append({
                "path": entry.path, "start_line": lo, "end_line": hi, "match_line": line,
                "language": entry.language, "score": round(score, 2),
                "snippet": "\n".join(f"{n:>5}: {entry.lines[n - 1]}" for n in range(lo, hi + 1)),
            })
        return results

    def outline(self, path: str | None = None) -> dict:
        if path:
            entry = self.entry(path)
            return {
                "path": entry.path, "language": entry.language, "lines": len(entry.lines),
                "imports": self.imports.get(entry.path, [])[:60],
                "symbols": [{"name": s.name, "kind": s.kind, "lines": f"{s.start}-{s.end}"}
                            for s in self.symbols if s.path == entry.path][:200],
                "syntax_error": entry.syntax_error,
            }
        folders = Counter(p.split("/", 1)[0] if "/" in p else "(root)" for p in self.files)
        modules = {}
        for symbol in self.symbols:
            if symbol.kind != "method":
                modules.setdefault(symbol.path, []).append(symbol.name)
        return {
            **self.summary(),
            "top_level": dict(folders.most_common(40)),
            "python_modules": {p: names[:15] for p, names in list(modules.items())[:60]},
            "dependency_files": [p for p in self.files if PurePosixPath(p).name.lower() in (
                "requirements.txt", "pyproject.toml", "setup.cfg", "setup.py", "pipfile",
                "package.json", "environment.yml", "poetry.lock", "go.mod", "cargo.toml")],
        }

    def trace(self, symbol: str, max_results: int = 30) -> dict:
        wanted = symbol.strip().split(".")[-1].strip("()")
        if not wanted:
            raise WorkspaceError("Name a function, method or class to trace.")
        definitions = [s for s in self.symbols if s.name.split(".")[-1] == wanted]
        callers = []
        for s in self.symbols:
            for name, line in s.calls:
                if name == wanted:
                    callers.append({"caller": s.name, "path": s.path, "line": line})
        module_calls = []
        for entry in self.files.values():
            if entry.language != "Python":
                continue
            for number, line in enumerate(entry.lines, start=1):
                if re.search(rf"\b{re.escape(wanted)}\s*\(", line) and not re.match(
                        r"\s*(?:async\s+)?def\s", line):
                    module_calls.append({"path": entry.path, "line": number,
                                         "code": line.strip()[:200]})
        return {
            "symbol": wanted,
            "definitions": [{"name": d.name, "kind": d.kind, "path": d.path,
                             "lines": f"{d.start}-{d.end}",
                             "calls": sorted({c for c, _ in d.calls})[:40]}
                            for d in definitions][:10],
            "called_from": callers[:max_results],
            "call_sites": module_calls[:max_results],
            "note": "Static analysis by name only: dynamic dispatch, aliases and calls through "
                    "variables are not resolved, so this list can be incomplete.",
        }

    def raw_text(self, path: str) -> str:
        """Current text on disk (NOT redacted) for patching; path re-validated."""
        entry = self.entry(path)
        target = safe_join(self.root, entry.path)
        raw = target.read_bytes()
        text = files.decode_text(raw)
        if text is None:
            raise WorkspaceError("That file is no longer a text file.")
        return text


# ------------------------------------------------------------------ helpers
def split_identifier(text: str) -> list[str]:
    out = []
    for token in _TOKEN.findall(text):
        for part in _CAMEL.sub(" ", token).replace("_", " ").split():
            out.append(part.lower())
        if "_" in token or _CAMEL.search(token):
            out.append(token.lower())
    return out


def query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(t for t in split_identifier(query)
                              if t not in _STOP and (len(t) > 1 or t.isdigit())))


def safe_join(root: Path, relative: str) -> Path:
    """root/relative, refusing anything that resolves outside root or through a symlink."""
    rel = PurePosixPath(relative.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or re.match(r"^[A-Za-z]:", relative):
        raise WorkspaceError("Path is outside the project.")
    root = root.resolve()
    target = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise WorkspaceError("Symbolic links are not followed.")
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise WorkspaceError("Path is outside the project.")
    return resolved


def _python_analysis(path: str, text: str) -> tuple[list[Symbol], list[str], str | None, bool]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        return [], [], f"{exc.__class__.__name__} at line {line}: {getattr(exc, 'msg', exc)}", \
            False
    symbols: list[Symbol] = []
    imports: list[str] = []
    is_main = False

    def call_name(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def visit(node, prefix: str, in_class: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = ("class" if isinstance(child, ast.ClassDef)
                        else "method" if in_class else "function")
                name = f"{prefix}{child.name}"
                symbol = Symbol(name, kind, path, child.lineno,
                                getattr(child, "end_lineno", child.lineno))
                if kind != "class":
                    symbol.calls = [(n, c.lineno) for c in ast.walk(child)
                                    if isinstance(c, ast.Call) and (n := call_name(c))][:300]
                symbols.append(symbol)
                visit(child, f"{name}.", kind == "class")

    visit(tree, "", False)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append("." * node.level + (node.module or ""))
        elif isinstance(node, ast.If) and _is_main_guard(node.test):
            is_main = True
    return symbols, sorted(set(imports)), None, is_main


def _is_main_guard(test: ast.expr) -> bool:
    """Exactly `__name__ == "__main__"` (either order)."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = [test.left, test.comparators[0]]
    names = {s.id for s in sides if isinstance(s, ast.Name)}
    values = {s.value for s in sides if isinstance(s, ast.Constant)}
    return names == {"__name__"} and values == {"__main__"}


def _entry_points(ws: Workspace, mains: set[str]) -> list[dict]:
    points = [{"path": p, "kind": "script", "evidence": "if __name__ == '__main__'"}
              for p in sorted(mains)]
    for path, entry in ws.files.items():
        name = PurePosixPath(path).name.lower()
        if entry.language == "Python" and "streamlit" in ws.imports.get(path, []) and (
                "st.set_page_config" in entry.text or "st.navigation" in entry.text):
            points.append({"path": path, "kind": "Streamlit app",
                           "evidence": "imports streamlit and configures pages"})
        elif name in ("manage.py", "wsgi.py", "asgi.py") and entry.language == "Python":
            points.append({"path": path, "kind": "web app", "evidence": name})
        elif name == "pyproject.toml":
            try:
                data = tomllib.loads(entry.text)
            except tomllib.TOMLDecodeError:
                continue
            for script, target in (data.get("project", {}).get("scripts", {}) or {}).items():
                points.append({"path": path, "kind": "console script",
                               "evidence": f"{script} = {target}"})
        elif name == "dockerfile":
            for line in entry.lines:
                if line.strip().upper().startswith(("CMD", "ENTRYPOINT")):
                    points.append({"path": path, "kind": "container", "evidence": line.strip()})
    seen, unique = set(), []
    for point in points:
        key = (point["path"], point["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _index(ws: Workspace, candidates: list[tuple[str, bytes]], limits: Limits) -> Workspace:
    total = 0
    mains: set[str] = set()
    for rel, raw in sorted(candidates):
        language, reason = files.classify(rel, len(raw), limits.max_file_bytes)
        if language is None:
            ws.skipped[reason] += 1
            continue
        if len(ws.files) >= limits.max_files:
            ws.skipped["file limit reached"] += 1
            ws.truncated = True
            continue
        text = files.decode_text(raw)
        if text is None:
            ws.skipped["binary content"] += 1
            continue
        if total + len(text) > limits.max_total_chars:
            ws.skipped["total size limit reached"] += 1
            ws.truncated = True
            continue
        total += len(text)
        redacted, count = files.prepare_text(text, language)
        entry = FileEntry(rel, language, redacted, redacted.splitlines(),
                          hashlib.sha256(raw).hexdigest(), count)
        if language == "Python":
            symbols, imports, error, is_main = _python_analysis(rel, text)
            ws.symbols.extend(symbols)
            ws._defs[rel] = {s.name.rsplit(".", 1)[-1].lower() for s in symbols}
            ws.imports[rel] = [i.split(".")[0] for i in imports]
            entry.syntax_error = error
            if is_main:
                mains.add(rel)
        ws.files[rel] = entry
        tokens = Counter(split_identifier(redacted))
        ws._tokens[rel] = tokens
        ws._df.update(set(tokens))
    ws.entry_points = _entry_points(ws, mains)
    return ws


# ---------------------------------------------------------------- ZIP uploads
def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def load_zip(raw: bytes, name: str, *, owner: str, dest_parent: Path,
             limits: Limits) -> Workspace:
    if len(raw) > limits.zip_max_bytes:
        raise WorkspaceError(f"The archive is larger than {limits.zip_max_bytes // 1_000_000} MB.")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise WorkspaceError("That file is not a valid ZIP archive.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        members = archive.infolist()
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise WorkspaceError("The ZIP archive is damaged and can't be read.") from exc
    if len(members) > limits.zip_max_members:
        raise WorkspaceError(f"The archive has {len(members):,} entries; the limit is "
                             f"{limits.zip_max_members:,}.")
    if sum(m.file_size for m in members) > limits.zip_max_uncompressed:
        raise WorkspaceError("The archive expands to more than "
                             f"{limits.zip_max_uncompressed // 1_000_000} MB.")
    ws_id = uuid.uuid4().hex
    dest = dest_parent / ws_id
    dest.mkdir(parents=True, exist_ok=False)
    workspace = Workspace(ws_id, owner, "upload", _clean_name(name), dest, trusted=False)
    names = [m.filename.replace("\\", "/") for m in members if not m.is_dir()]
    prefix = _common_prefix(names)
    candidates: list[tuple[str, bytes]] = []
    try:
        for info in members:
            if info.is_dir():
                continue
            original = info.filename.replace("\\", "/")
            if (original.startswith("/") or re.match(r"^[A-Za-z]:", original)
                    or ".." in PurePosixPath(original).parts):
                raise WorkspaceError(f"The archive contains an unsafe path ({original[:80]}); "
                                     "nothing was extracted.")
            if _is_symlink(info):
                workspace.skipped["symbolic link"] += 1
                continue
            if info.flag_bits & 0x1:
                workspace.skipped["encrypted entry"] += 1
                continue
            rel = original[len(prefix):] if prefix else original
            language, reason = files.classify(rel, info.file_size, limits.max_file_bytes)
            if language is None:
                workspace.skipped[reason] += 1
                continue
            if info.compress_size and info.file_size / info.compress_size > 200 \
                    and info.file_size > 1_000_000:
                raise WorkspaceError("The archive looks like a compression bomb; nothing was "
                                     "extracted.")
            with archive.open(info) as handle:
                data = handle.read(limits.max_file_bytes + 1)
            if len(data) > limits.max_file_bytes:
                workspace.skipped["file too large"] += 1
                continue
            target = safe_join(dest, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            candidates.append((rel, data))
    except WorkspaceError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise WorkspaceError("The ZIP archive is damaged or uses an unsupported format.") \
            from exc
    finally:
        archive.close()
    if not candidates:
        shutil.rmtree(dest, ignore_errors=True)
        raise WorkspaceError("The archive contains no supported source or text files.")
    return _index(workspace, candidates, limits)


def _common_prefix(names: list[str]) -> str:
    """Strip a single top folder (GitHub 'Download ZIP' style)."""
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and all("/" in n for n in names):
        return next(iter(tops)) + "/"
    return ""


def _clean_name(name: str) -> str:
    base = PurePosixPath((name or "project").replace("\\", "/")).name
    return re.sub(r"[^\w.\- ]+", "_", base)[:80] or "project"


# -------------------------------------------------------------- local folders
def validate_local_root(path_text: str) -> Path:
    if not path_text or not path_text.strip():
        raise WorkspaceError("Enter the full path of a project folder.")
    candidate = Path(path_text.strip().strip("\"'")).expanduser()
    if not candidate.is_absolute():
        raise WorkspaceError("Use a full (absolute) folder path.")
    if candidate.is_symlink():
        raise WorkspaceError("Choose the real folder, not a symbolic link.")
    root = candidate.resolve()
    if not root.is_dir():
        raise WorkspaceError("That folder doesn't exist.")
    home = Path.home().resolve()
    if root.parent == root or root == home or root in home.parents:
        raise WorkspaceError("That folder is too broad. Choose a specific project folder.")
    lowered = {p.lower() for p in root.parts}
    if lowered & {".ssh", ".aws", ".gnupg", "windows", "system32", "program files"}:
        raise WorkspaceError("That folder isn't a project folder that may be inspected.")
    return root


def load_local(path_text: str, *, owner: str, limits: Limits, trusted_roots: list[Path],
               exclude_dirs: list[Path] | None = None) -> Workspace:
    """`exclude_dirs`: app-owned folders (extracted uploads, model cache) never indexed."""
    root = validate_local_root(path_text)
    excluded = {d.resolve() for d in (exclude_dirs or [])}
    trusted = any(root == t or t in root.parents for t in trusted_roots)
    workspace = Workspace(uuid.uuid4().hex, owner, "local", root.name, root, trusted=trusted)
    candidates: list[tuple[str, bytes]] = []
    seen = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        here = Path(current)
        kept = []
        for d in dirs:
            rel_dir = (here / d).relative_to(root).as_posix()
            if (here / d).is_symlink():
                workspace.skipped["symbolic link"] += 1
            elif (here / d).resolve() in excluded:
                workspace.skipped["app storage folder"] += 1
            elif files.classify(rel_dir + "/x.py", 0, 1)[1] == "excluded folder":
                workspace.skipped["excluded folder"] += 1
            else:
                kept.append(d)
        dirs[:] = sorted(kept)
        for name in sorted(names):
            seen += 1
            if seen > limits.max_files * 20:
                workspace.truncated = True
                break
            full = here / name
            rel = full.relative_to(root).as_posix()
            if full.is_symlink():
                workspace.skipped["symbolic link"] += 1
                continue
            try:
                size = full.stat().st_size
            except OSError:
                workspace.skipped["unreadable"] += 1
                continue
            language, reason = files.classify(rel, size, limits.max_file_bytes)
            if language is None:
                workspace.skipped[reason] += 1
                continue
            try:
                resolved = full.resolve()
                if root not in resolved.parents:
                    workspace.skipped["outside the project"] += 1
                    continue
                candidates.append((rel, resolved.read_bytes()[: limits.max_file_bytes + 1]))
            except OSError:
                workspace.skipped["unreadable"] += 1
    if not candidates:
        raise WorkspaceError("No supported source or text files were found in that folder.")
    return _index(workspace, candidates, limits)
