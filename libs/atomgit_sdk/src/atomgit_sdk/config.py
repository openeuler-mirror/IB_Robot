"""
AtomGit SDK Configuration Management
"""

import json
import os
import re
from typing import Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field
from atomgit_sdk.exceptions import ConfigurationError
from atomgit_sdk.utils.url import parse_atomgit_url


def _expand_env_var(value: str) -> str:
    """
    Expand environment variable in value.
    Supports $VAR_NAME or ${VAR_NAME} format.
    """
    if not value:
        return value

    env_pattern = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

    def replace_env(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ConfigurationError(
                f"Environment variable '{var_name}' is not set. "
                f"Please add 'export {var_name}=your_token' to your ~/.bashrc or ~/.zshrc"
            )
        return env_value

    return env_pattern.sub(replace_env, value)


class AtomGitConfig(BaseModel):
    """AtomGit API Configuration"""

    token: str = Field(..., description="AtomGit API token")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    base_url: str = Field(
        default="https://api.atomgit.com", description="AtomGit API base URL"
    )

    class Config:
        frozen = True

    def with_overrides(
        self,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "AtomGitConfig":
        """Return a new config with selected fields overridden."""
        return type(self)(
            token=self.token,
            owner=owner or self.owner,
            repo=repo or self.repo,
            base_url=base_url or self.base_url,
        )

    @classmethod
    def from_json(
        cls,
        config_path: str = "config.json",
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        url: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "AtomGitConfig":
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            AtomGitConfig instance

        Raises:
            ConfigurationError: If required fields are missing
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")

        atomgit_section = config.get("atomgit", {})
        parsed_url = parse_atomgit_url(url) if url else {}

        token = atomgit_section.get("token")
        resolved_owner = owner or parsed_url.get("owner") or atomgit_section.get("owner")
        resolved_repo = repo or parsed_url.get("repo") or atomgit_section.get("repo")

        if not all([token, resolved_owner, resolved_repo]):
            missing = []
            if not token:
                missing.append("token")
            if not resolved_owner:
                missing.append("owner")
            if not resolved_repo:
                missing.append("repo")
            raise ConfigurationError(
                f"Missing required configuration fields: {', '.join(missing)}"
            )

        token = _expand_env_var(token)

        return cls(
            token=token,
            owner=resolved_owner,
            repo=resolved_repo,
            base_url=base_url
            or atomgit_section.get("baseUrl", "https://api.atomgit.com"),
        )


def resolve_atomgit_context(
    config_path: str = "config.json",
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    url: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[AtomGitConfig, Dict[str, Union[str, int]]]:
    """Resolve effective AtomGit config plus parsed URL metadata."""
    parsed_url = parse_atomgit_url(url) if url else {}
    config = AtomGitConfig.from_json(
        config_path,
        owner=owner or parsed_url.get("owner"),
        repo=repo or parsed_url.get("repo"),
        base_url=base_url,
    )
    return config, parsed_url
