"""Fresh interpreters keep other test imports from hiding eager dependencies."""

import os
import subprocess
import sys
from pathlib import Path


def test_disabled_package_import_only_loads_instrumentation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import logging
import sys
from pathlib import Path

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith(('lttngust', 'ibrobot_tracing.'))
                and fullname != 'ibrobot_tracing.instrumentation'):
            raise AssertionError('Unexpected import: ' + fullname)

logging.root.setLevel(logging.ERROR)
sys.meta_path.insert(0, BlockHeavyImports())
import ibrobot_tracing as trace
assert Path(trace.__file__).resolve() == Path(sys.argv[1])
assert logging.root.level == logging.ERROR
assert set(name for name in sys.modules if name.startswith('ibrobot_tracing.')) == {
    'ibrobot_tracing.instrumentation',
}
assert 'lttngust' not in sys.modules
assert set(trace.__all__) <= set(dir(trace))
assert not trace.get_trace_emitter('business').enabled
with trace.span('disabled', status='collision', timestamp_ns=123):
    trace.mark('disabled')
assert trace.start_span('disabled') is None
trace.end_span(None)
""",
            str(Path(__file__).resolve().parents[1] / "ibrobot_tracing" / "__init__.py"),
        ],
        env={**os.environ, "IB_TRACE_ENABLED": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_exports_preserve_public_api_identity():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import ibrobot_tracing as trace
for name, module_name in trace._LAZY_EXPORTS.items():
    value = getattr(trace, name)
    assert value is getattr(importlib.import_module('ibrobot_tracing.' + module_name), name)
    assert value is trace.__dict__[name]
namespace = {}
exec('from ibrobot_tracing import *', namespace)
assert set(trace.__all__) <= namespace.keys()
try:
    trace.no_such_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export must fail')
""",
        ],
        env={**os.environ, "IB_TRACE_ENABLED": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
