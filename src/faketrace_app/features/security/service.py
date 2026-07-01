from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ScanResult:
    key: str
    title: str
    ready: bool
    status: str
    tool: str
    target: str
    summary: str
    command: list[str]
    issues: list[dict[str, Any]]
    issue_count: int
    raw_output: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "ready": self.ready,
            "status": self.status,
            "tool": self.tool,
            "target": self.target,
            "summary": self.summary,
            "command": self.command,
            "issues": self.issues,
            "issue_count": self.issue_count,
            "raw_output": self.raw_output,
            "error": self.error,
        }


class SecurityScanService:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[4]

    def run_all(self) -> dict[str, Any]:
        dependency_scan = self._run_dependency_scan()
        static_scan = self._run_static_scan()
        scans = [dependency_scan, static_scan]
        ready_count = sum(1 for scan in scans if scan.ready)
        risk_count = sum(scan.issue_count for scan in scans)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "scan_count": len(scans),
                "ready_count": ready_count,
                "risk_count": risk_count,
            },
            "results": [scan.to_dict() for scan in scans],
        }

    def _run_dependency_scan(self) -> ScanResult:
        requirements = self.root_dir / "requirements.txt"
        command = ["pip-audit", "-r", str(requirements), "-f", "json"]
        if not requirements.exists():
            return self._missing_target_result(
                key="dependency",
                title="项目依赖安全扫描",
                tool="pip-audit",
                target=str(requirements),
                command=command,
                message="未找到 requirements.txt，无法执行依赖扫描。",
            )

        return self._run_scan(
            key="dependency",
            title="项目依赖安全扫描",
            tool="pip-audit",
            target=str(requirements.relative_to(self.root_dir)),
            command=command,
            parser=self._parse_pip_audit,
        )

    def _run_static_scan(self) -> ScanResult:
        scan_targets = [path for path in ["src", "app.py", "web_app.py"] if (self.root_dir / path).exists()]
        command = ["bandit", "-r", *scan_targets, "-f", "json"]
        if not scan_targets:
            return self._missing_target_result(
                key="static",
                title="代码仓库静态扫描",
                tool="bandit",
                target="src app.py web_app.py",
                command=command,
                message="未找到可扫描的代码目录或入口文件。",
            )

        return self._run_scan(
            key="static",
            title="代码仓库静态扫描",
            tool="bandit",
            target=", ".join(scan_targets),
            command=command,
            parser=self._parse_bandit,
        )

    def _run_scan(
        self,
        *,
        key: str,
        title: str,
        tool: str,
        target: str,
        command: list[str],
        parser,
    ) -> ScanResult:
        tool_path = shutil.which(command[0])
        if tool_path is None:
            return ScanResult(
                key=key,
                title=title,
                ready=False,
                status="tool_missing",
                tool=tool,
                target=target,
                summary=f"未检测到 {tool}，请先安装对应工具后再执行扫描。",
                command=command,
                issues=[],
                issue_count=0,
                raw_output="",
                error=f"缺少命令：{tool}。建议安装：pip install {tool}",
            )

        safe_command = [tool_path, *command[1:]]
        completed = subprocess.run(
            safe_command,  # nosec B603
            cwd=self.root_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raw_output = stdout or stderr
        parser_input = stdout or stderr

        try:
            summary, issues, issue_count = parser(parser_input)
            status = "issues_found" if issue_count else "clean"
            error = stderr or None
        except Exception as exc:
            summary = f"{tool} 已执行，但结果解析失败，请查看原始输出。"
            issues = []
            issue_count = 0
            status = "parse_error"
            error = stderr or str(exc)

        if completed.returncode not in (0, 1):
            status = "command_error"
            summary = f"{tool} 执行失败，请检查工具安装或项目依赖。"
            error = stderr or f"命令退出码：{completed.returncode}"

        return ScanResult(
            key=key,
            title=title,
            ready=True,
            status=status,
            tool=tool,
            target=target,
            summary=summary,
            command=safe_command,
            issues=issues,
            issue_count=issue_count,
            raw_output=raw_output,
            error=error,
        )

    def _missing_target_result(
        self,
        *,
        key: str,
        title: str,
        tool: str,
        target: str,
        command: list[str],
        message: str,
    ) -> ScanResult:
        return ScanResult(
            key=key,
            title=title,
            ready=False,
            status="target_missing",
            tool=tool,
            target=target,
            summary=message,
            command=command,
            issues=[],
            issue_count=0,
            raw_output="",
            error=message,
        )

    def _parse_pip_audit(self, stdout: str) -> tuple[str, list[dict[str, Any]], int]:
        text = (stdout or "").strip()
        if not text or "No known vulnerabilities found" in text:
            return "未发现已知依赖漏洞。", [], 0

        payload = self._load_json_payload(text, default=[])
        if isinstance(payload, dict):
            packages = payload.get("dependencies", [])
        else:
            packages = payload

        issues: list[dict[str, Any]] = []
        for package in packages:
            for vulnerability in package.get("vulns", []):
                fix_versions = vulnerability.get("fix_versions") or []
                issues.append(
                    {
                        "package": package.get("name", "unknown"),
                        "installed_version": package.get("version", "unknown"),
                        "id": vulnerability.get("id", "N/A"),
                        "severity": vulnerability.get("severity") or "unknown",
                        "description": vulnerability.get("description") or "未提供漏洞描述。",
                        "fix_versions": fix_versions,
                    }
                )

        issue_count = len(issues)
        if issue_count:
            summary = f"发现 {issue_count} 个依赖漏洞，建议根据修复版本尽快升级。"
        else:
            summary = "未发现已知依赖漏洞。"
        return summary, issues, issue_count

    def _parse_bandit(self, stdout: str) -> tuple[str, list[dict[str, Any]], int]:
        payload = self._load_json_payload(stdout or "{}", default={})
        findings = payload.get("results", [])
        issues = [
            {
                "file": item.get("filename", "unknown"),
                "line": item.get("line_number", 0),
                "test_id": item.get("test_id", "N/A"),
                "severity": item.get("issue_severity", "UNKNOWN"),
                "confidence": item.get("issue_confidence", "UNKNOWN"),
                "description": item.get("issue_text", "未提供问题描述。"),
            }
            for item in findings
        ]
        issue_count = len(issues)
        if issue_count:
            summary = f"发现 {issue_count} 个静态扫描告警，请结合严重级别逐项确认。"
        else:
            summary = "未发现静态扫描告警。"
        return summary, issues, issue_count

    def _load_json_payload(self, text: str, default):
        content = (text or "").strip()
        if not content:
            return default

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        for opener, closer in (("[", "]"), ("{", "}")):
            start = content.find(opener)
            end = content.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidate = content[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        raise json.JSONDecodeError("Unable to locate JSON payload.", content, 0)
