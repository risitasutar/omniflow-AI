"""Deterministic static security review. Nothing is executed or imported.

Python files are analysed with the `ast` module (with a small same-function taint check:
values that come from `request.*`, `input()` or `sys.argv` raise confidence when they reach
a dangerous call). All files are checked for hardcoded secrets. Evidence lines are taken
from the redacted text, so secret values never appear in a finding.

Confidence levels:
- confirmed: the dangerous construct is present and its risky argument is visible.
- suspicious: dangerous if the data is attacker-controlled, which can't be proven statically.
- manual_review: often fine; a person should check the context.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

from omniflow.code import files
from omniflow.code.workspace import Workspace

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_SECRET_NAME = re.compile(r"(?i)(pass(word|wd)?|pwd|secret|token|api_?key|apikey|"
                          r"access_?key|private_?key|client_?secret|auth_?key|signing_?key)")
_PLACEHOLDER = re.compile(r"(?i)^(|<.*>|\$\{.*\}|your[-_ ].*|change[-_ ]?me|x+|\*+|"
                          r"example.*|placeholder|todo|test.*|dummy.*|fake.*|none|null|"
                          r"redacted|«redacted»|.{0,7})$")
# Variables that hold the *name* of a secret, not the secret itself.
_NOT_A_SECRET_NAME = re.compile(r"(?i)(_name|_names|_header|_field|_env|_var|_param|_prefix|"
                                r"_url|_path|_file|_label|_pattern|_regex|_re)$")
# Values shaped like header names ("X-Api-Key") or env var names ("API_KEY").
_NAME_LIKE_VALUE = re.compile(r"^(?:[A-Za-z]+(?:-[A-Za-z]+)+|[A-Z][A-Z0-9_]*)$")
_SQL_WORDS = re.compile(r"(?i)\b(select|insert|update|delete|drop|create|alter|replace)\b"
                        r".*\b(from|into|table|set|where|values)\b")
_TOKEN_PATTERNS = [
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]
_JS_RULES = [
    ("JS001", r"\beval\s*\(", "Dynamic code execution (eval)", "high", "suspicious",
     "Runs arbitrary code if the argument is attacker-controlled.",
     "Avoid eval; parse data with JSON.parse or use explicit logic."),
    ("JS002", r"\.innerHTML\s*=(?!=)", "Assignment to innerHTML", "medium", "suspicious",
     "Cross-site scripting if the value contains user input.",
     "Use textContent, or sanitise HTML with a vetted library."),
    ("JS003", r"child_process\.(exec|execSync)\s*\(", "Shell command execution", "high",
     "suspicious", "Command injection if the command includes user input.",
     "Use execFile/spawn with an argument array and no shell."),
]


@dataclass(frozen=True)
class Finding:
    rule: str
    title: str
    severity: str  # high | medium | low
    confidence: str  # confirmed | suspicious | manual_review
    path: str
    line: int
    evidence: str
    impact: str
    remediation: str

    def as_dict(self) -> dict:
        return asdict(self)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _is_literal(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_literal(e) for e in node.elts)
    return False


def _is_dynamic_string(node: ast.AST) -> bool:
    """f-string, % formatting, .format() or + concatenation with a non-literal part."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return not (_is_literal(node.left) and _is_literal(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "format":
        return True
    return False


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


class _PythonScanner(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]):
        self.path, self.lines = path, lines
        self.findings: list[Finding] = []
        self.tainted: set[str] = set()
        self.sql_strings: dict[str, int] = {}

    # ------------------------------------------------------------ helpers
    def add(self, node, rule, title, severity, confidence, impact, remediation):
        line = getattr(node, "lineno", 1)
        text = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(rule, title, severity, confidence, self.path, line,
                                     text[:220], impact, remediation))

    def is_tainted(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in self.tainted:
                return True
            dotted = _dotted(sub)
            if dotted.startswith(("request.", "flask.request")) or dotted in (
                    "input", "sys.argv"):
                return True
        return False

    def visit_FunctionDef(self, node):  # noqa: N802 - ast visitor naming
        saved = (self.tainted, self.sql_strings)
        self.tainted, self.sql_strings = set(saved[0]), {}
        self.generic_visit(node)
        self.tainted, self.sql_strings = saved

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Assign(self, node):  # noqa: N802
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if self.is_tainted(node.value):
            self.tainted.update(targets)
        if _is_dynamic_string(node.value) and _SQL_WORDS.search(ast.unparse(node.value)):
            for name in targets:
                self.sql_strings[name] = node.lineno
        for target in node.targets:
            name = target.id if isinstance(target, ast.Name) else (
                target.attr if isinstance(target, ast.Attribute) else "")
            value = node.value
            if (name and _SECRET_NAME.search(name) and not _NOT_A_SECRET_NAME.search(name)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str) and len(value.value) >= 8
                    and not _PLACEHOLDER.match(value.value.strip())
                    and not _NAME_LIKE_VALUE.match(value.value.strip())
                    and " " not in value.value.strip()):
                self.add(node, "PY007", f"Hardcoded secret in '{name}'", "high", "suspicious",
                         "Anyone with the source code gets the credential.",
                         "Load it from an environment variable or a secrets manager, and "
                         "rotate the exposed value.")
        self.generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        name = _dotted(node.func)
        short = name.rsplit(".", 1)[-1]
        first = node.args[0] if node.args else None

        if name in ("eval", "exec") or (name == "compile" and first is not None
                                         and not _is_literal(first)):
            if not _is_literal(first):
                self.add(node, "PY001", f"Dynamic code execution ({name})", "high",
                         "confirmed" if self.is_tainted(first) else "suspicious",
                         "Runs arbitrary code if the input is attacker-controlled.",
                         "Avoid eval/exec; use ast.literal_eval for literals or explicit "
                         "parsing.")

        if name.startswith("subprocess.") and short in ("getoutput", "getstatusoutput") \
                and first is not None and not _is_literal(first):
            self.add(node, "PY002", f"Shell command via {name}", "high",
                     "confirmed" if self.is_tainted(first) else "suspicious",
                     "These helpers always use a shell: command injection if any part of "
                     "the command comes from user input.",
                     "Use subprocess.run with an argument list (no shell).")
        if name.startswith("subprocess.") and short in ("Popen", "run", "call", "check_call",
                                                        "check_output"):
            if _is_true(_kw(node, "shell")):
                dynamic = first is not None and not _is_literal(first)
                self.add(node, "PY002", "Shell command with shell=True",
                         "high" if dynamic else "medium",
                         "confirmed" if self.is_tainted(first) else (
                             "suspicious" if dynamic else "manual_review"),
                         "Command injection if any part of the command comes from user input.",
                         "Pass an argument list and remove shell=True.")
        if name in ("os.system", "os.popen") and first is not None and not _is_literal(first):
            self.add(node, "PY002", f"Shell command via {name}", "high",
                     "confirmed" if self.is_tainted(first) else "suspicious",
                     "Command injection if any part of the command comes from user input.",
                     "Use subprocess.run with an argument list (no shell).")

        if short in ("execute", "executemany", "executescript", "read_sql", "read_sql_query",
                     "raw", "text") and first is not None:
            sql_dynamic = (_is_dynamic_string(first) and _SQL_WORDS.search(ast.unparse(first)))
            via_variable = isinstance(first, ast.Name) and first.id in self.sql_strings
            if sql_dynamic or via_variable:
                self.add(node, "PY003", "SQL built with string formatting", "high",
                         "confirmed" if self.is_tainted(first) or (
                             via_variable and first.id in self.tainted) else "suspicious",
                         "SQL injection: an attacker could read or change the database.",
                         "Use parameterised queries (placeholders plus a parameters tuple).")

        if name in ("pickle.load", "pickle.loads", "cPickle.loads", "marshal.loads",
                    "marshal.load", "dill.loads", "dill.load", "shelve.open",
                    "jsonpickle.decode"):
            self.add(node, "PY004", f"Insecure deserialisation ({name})", "high", "suspicious",
                     "Loading attacker-controlled data executes arbitrary code.",
                     "Use a data-only format such as JSON for untrusted input.")
        if name in ("yaml.load", "yaml.load_all"):
            loader = _kw(node, "Loader") or (node.args[1] if len(node.args) > 1 else None)
            if loader is None or "Safe" not in _dotted(loader):
                self.add(node, "PY004", "yaml.load without SafeLoader", "high", "confirmed",
                         "YAML tags can construct arbitrary Python objects.",
                         "Use yaml.safe_load (or Loader=yaml.SafeLoader).")
        if name == "yaml.unsafe_load":
            self.add(node, "PY004", "yaml.unsafe_load", "high", "confirmed",
                     "YAML tags can construct arbitrary Python objects.", "Use yaml.safe_load.")

        if _is_false(_kw(node, "verify")):  # requests, httpx, sessions and clients alike
            self.add(node, "PY005", "TLS certificate verification disabled", "medium",
                     "confirmed", "Traffic can be intercepted and modified (MITM).",
                     "Remove verify=False; point verify at a CA bundle if needed.")
        if name.startswith("requests.") and short in (
                "get", "post", "put", "delete", "patch", "head", "request") \
                and _kw(node, "timeout") is None:
            self.add(node, "PY005", "HTTP request without a timeout", "low",
                     "manual_review", "A slow or hostile server can hang the program.",
                     "Pass timeout=(connect, read) seconds.")
        if name in ("ssl._create_unverified_context",) or (
                short == "wrap_socket" and _dotted(_kw(node, "cert_reqs") or ast.Name("x"))
                .endswith("CERT_NONE")):
            self.add(node, "PY005", "Unverified TLS context", "medium", "confirmed",
                     "Certificates aren't checked, so connections can be intercepted.",
                     "Use ssl.create_default_context().")

        if name == "tempfile.mktemp":
            self.add(node, "PY006", "Insecure temporary file (tempfile.mktemp)", "medium",
                     "confirmed", "Race condition: another process can create the file first.",
                     "Use tempfile.NamedTemporaryFile or tempfile.mkstemp.")

        if name in ("open", "io.open", "send_file", "flask.send_file", "os.remove",
                    "os.unlink", "shutil.rmtree", "Path") and self.is_tainted(first):
            self.add(node, "PY008", f"File path from user input ({name})", "high", "suspicious",
                     "Path traversal: '../' sequences can reach files outside the intended "
                     "folder.",
                     "Resolve the path and check it stays inside an allowed base folder; use "
                     "werkzeug.utils.secure_filename for uploaded names.")
        if short == "extractall" and _kw(node, "filter") is None and not node.args:
            self.add(node, "PY008", "Archive extracted without member checks", "medium",
                     "suspicious", "Zip/Tar slip: archive entries with '../' can overwrite "
                     "files outside the target folder.",
                     "Validate each member path (or use tarfile's filter='data').")

        if short == "run" and _is_true(_kw(node, "debug")):
            self.add(node, "PY009", "Web app started with debug=True", "medium", "confirmed",
                     "The debugger allows remote code execution if reachable.",
                     "Disable debug outside local development.")
        if short == "Environment" and _is_false(_kw(node, "autoescape")):
            self.add(node, "PY011", "Jinja2 autoescape disabled", "medium", "confirmed",
                     "Cross-site scripting through unescaped template variables.",
                     "Use autoescape=select_autoescape() or True.")
        if name in ("hashlib.md5", "hashlib.sha1"):
            self.add(node, "PY010", f"Weak hash ({name})", "low", "manual_review",
                     "Unsafe for passwords or signatures; fine for non-security checksums.",
                     "Use hashlib.sha256 for integrity, or bcrypt/argon2/scrypt for "
                     "passwords.")
        self.generic_visit(node)


def _line_findings(path: str, raw_lines: list[str], shown_lines: list[str],
                   language: str) -> list[Finding]:
    found = []
    for number, line in enumerate(raw_lines, start=1):
        for label, pattern in _TOKEN_PATTERNS:
            if pattern.search(line):
                evidence = shown_lines[number - 1] if number <= len(shown_lines) else ""
                found.append(Finding("SEC001", f"{label} in source", "high", "confirmed", path,
                                     number, evidence.strip()[:220],
                                     "A live credential in the repository can be used by "
                                     "anyone who can read it.",
                                     "Revoke and rotate it now, remove it from the code and "
                                     "Git history, and load it from the environment."))
                break
        if language in ("JavaScript", "TypeScript", "Vue", "Svelte", "HTML"):
            for rule, regex, title, severity, confidence, impact, fix in _JS_RULES:
                if re.search(regex, line):
                    found.append(Finding(rule, title, severity, confidence, path, number,
                                         shown_lines[number - 1].strip()[:220], impact, fix))
    return found


def scan(workspace: Workspace, path: str | None = None, max_findings: int = 200) -> dict:
    targets = [workspace.entry(path)] if path else list(workspace.files.values())
    findings: list[Finding] = []
    scanned = 0
    for entry in targets:
        try:
            raw = workspace.raw_text(entry.path)
        except (OSError, ValueError):
            continue
        scanned += 1
        raw_lines = raw.splitlines()
        findings.extend(_line_findings(entry.path, raw_lines, entry.lines, entry.language))
        if entry.language == "Python":
            try:
                tree = ast.parse(raw)
            except (SyntaxError, ValueError):
                continue
            scanner = _PythonScanner(entry.path, entry.lines)
            scanner.visit(tree)
            findings.extend(scanner.findings)
    unique = {(f.path, f.line, f.rule): f for f in findings}
    ordered = sorted(unique.values(), key=lambda f: (SEVERITY_ORDER[f.severity], f.path,
                                                     f.line))
    for finding in ordered:  # belt and braces: no secret value in any evidence line
        if files.looks_secret(finding.evidence):
            object.__setattr__(finding, "evidence", "(line hidden: contains a secret value)")
    counts = {s: sum(f.severity == s for f in ordered) for s in SEVERITY_ORDER}
    return {
        "files_scanned": scanned,
        "counts": counts,
        "findings": [f.as_dict() for f in ordered[:max_findings]],
        "truncated": len(ordered) > max_findings,
        "note": "Deterministic pattern checks only. No findings does not mean the code is "
                "secure; logic flaws, authentication and authorisation issues, and "
                "vulnerable dependencies need a manual review.",
    }
