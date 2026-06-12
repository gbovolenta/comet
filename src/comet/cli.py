"""Command-line interface for the comet package."""

from __future__ import annotations

import argparse
import sys

from comet.workflows import run as workflow_run


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser for `comet`.

    Returns:
        argparse.ArgumentParser: Configured parser with the available
        subcommands and arguments.
    """
    parser = argparse.ArgumentParser(
        prog="comet",
        description="Constant pressure controller using the Metropolis algorithm",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a Constant-Pressure simulations workflow from a YAML configuration file"
    )
    run_parser.add_argument(
        "config",
        metavar="CONFIG",
        help="Path to the YAML configuration file",
    )

    run_parser.set_defaults(func=_handle_run)

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    """Dispatch the parsed `run` subcommand arguments to the workflow.

    Args:
        args: Parsed command-line namespace containing a `config` path.

    Returns:
        int: Workflow exit status.
    """
    config_path = args.config
    #return workflow_run.run(config_path)
    return workflow_run(config_path)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested command.

    Args:
        argv: Optional argument vector. When omitted, arguments are read from
            `sys.argv`.

    Returns:
        int: Process-style exit status code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        try:
            return int(args.func(args))
        except NotImplementedError as exc:  # pragma: no cover - explicit user feedback
            parser.exit(status=1, message=f"{exc}\n")

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
