import sys
from typing import Literal

import rich


def report(kind: Literal["error", "warning", "info"], message: str) -> None:
    """Report a message to the console with appropriate styling."""

    prefix = {
        "error": "[bold red]ERROR[/bold red]",
        "warning": "[bold yellow]WARNING[/bold yellow]",
        "info": "[bold green]INFO[/bold green]",
    }[kind]

    rich.print(f"{prefix} {message}", file=sys.stderr)


def line(title: str = "") -> None:
    """Print a header for an example section."""

    line = "─" * 80
    rich.print(f"{line}", file=sys.stderr)

    if title:
        rich.print(f"[bold underline]{title}", file=sys.stderr)
