"""``$(find pkg)`` / ``$(env VAR)`` path substitution (profile/argument paths)."""

from __future__ import annotations

import os
import re

from ament_index_python.packages import get_package_share_directory

_FIND_RE = re.compile(r"\$\(find\s+([A-Za-z0-9_]+)\)")
_ENV_RE = re.compile(r"\$\(env\s+([A-Za-z_][A-Za-z0-9_]*)\)")


def resolve_path(value: str) -> str:
    """Expand ``$(find pkg)`` and ``$(env VAR)`` substitutions in a path."""

    def _find(match: re.Match) -> str:
        return get_package_share_directory(match.group(1))

    def _env(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"path references unset environment variable {name!r}: {value}")
        return os.environ[name]

    return _ENV_RE.sub(_env, _FIND_RE.sub(_find, str(value)))
