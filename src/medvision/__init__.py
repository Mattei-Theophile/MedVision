"""MedVision : commande domotique par gestes et langue des signes, 100 % locale."""

__version__ = "0.1.0"


def main() -> None:
    from medvision.cli import main as cli_main

    cli_main()
