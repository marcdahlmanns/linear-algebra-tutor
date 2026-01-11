"""Test CLI commands"""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def test(
    name: str = typer.Argument(..., help="Name"),
    count: int = typer.Option(5, help="Count"),
):
    """Test command"""
    console.print(f"Hello {name}, count={count}")

if __name__ == "__main__":
    app()
