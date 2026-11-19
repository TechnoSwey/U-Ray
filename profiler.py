"""Lightweight profiling for calls made through U-Ray's nn module."""

import json


def _shape_of(value, depth=0):
    if depth >= 6:
        return ["…"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        child_shape = _shape_of(value[0], depth + 1)
        return [len(value)] + (child_shape or [])
    shape = getattr(value, "shape", None)
    if callable(shape):
        try:
            return list(shape())
        except Exception:
            return None
    return None


def _describe(value):
    value_type = type(value).__name__
    shape = _shape_of(value)
    if shape is not None:
        return f"{value_type}{shape}"
    if isinstance(value, dict):
        return f"{value_type}(keys={len(value)})"
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value_type
    try:
        length = len(value)
    except (TypeError, AttributeError):
        return value_type
    return f"{value_type}(len={length})"


class MLProfiler:
    """Collect aggregate timings for nn.* calls without changing their results."""

    def __init__(self):
        self.operations = {}

    def record(self, name, elapsed_ns, args, result=None, failed=False):
        operation = self.operations.get(name)
        if operation is None:
            operation = {
                "name": name,
                "calls": 0,
                "failures": 0,
                "total_ns": 0,
                "max_ns": 0,
                "args": [_describe(value) for value in args],
                "result": _describe(result) if not failed else None,
            }
            self.operations[name] = operation
        operation["calls"] += 1
        operation["failures"] += int(failed)
        operation["total_ns"] += elapsed_ns
        operation["max_ns"] = max(operation["max_ns"], elapsed_ns)

    def report(self, backend, runtime_ns, program_output=None):
        operations = []
        for operation in self.operations.values():
            calls = operation["calls"]
            operations.append({
                "name": operation["name"],
                "calls": calls,
                "failures": operation["failures"],
                "total_seconds": operation["total_ns"] / 1_000_000_000,
                "average_seconds": (
                    operation["total_ns"] / calls / 1_000_000_000
                ),
                "max_seconds": operation["max_ns"] / 1_000_000_000,
                "args": operation["args"],
                "result": operation["result"],
            })
        operations.sort(key=lambda item: (-item["total_seconds"], item["name"]))
        report = {
            "backend": backend,
            "runtime_seconds": runtime_ns / 1_000_000_000,
            "profiled_operations": len(operations),
            "operations": operations,
        }
        if program_output is not None:
            report["program_output"] = program_output
        return report

    @staticmethod
    def format_text(report):
        lines = [
            "U-Ray ML profile",
            f"backend: {report['backend']}",
            f"program runtime: {report['runtime_seconds']:.6f}s",
        ]
        operations = report["operations"]
        if not operations:
            lines.append("nn.* calls: none")
            return "\n".join(lines)

        lines.append(
            f"{'operation':34} {'calls':>7} {'failed':>7} {'total':>10} "
            f"{'average':>10} {'max':>10}"
        )
        for operation in operations:
            name = operation["name"]
            if len(name) > 34:
                name = "…" + name[-33:]
            lines.append(
                f"{name:34} {operation['calls']:7d} "
                f"{operation['failures']:7d} "
                f"{operation['total_seconds']:9.6f}s "
                f"{operation['average_seconds']:9.6f}s "
                f"{operation['max_seconds']:9.6f}s"
            )
            lines.append(
                f"  args: {', '.join(operation['args']) or '(none)'}; "
                f"result: {operation['result'] or 'unavailable'}"
            )
        lines.append(
            "Times are inclusive per nn.* call; nested Python work inside "
            "a trainer is included in that trainer's row."
        )
        return "\n".join(lines)

    @staticmethod
    def format_json(report):
        return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
