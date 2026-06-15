import sys
import os

if sys.platform == "win32":
    os.system("")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt

from config import ALL_MODELS, DEFAULT_DEBATE_MODELS, DEFAULT_GROUP_A, DEFAULT_GROUP_B
from debate import roundtable_debate, group_debate, _model_label

console = Console()


def print_banner():
    banner = r"""
[bold bright_yellow]╔══════════════════════════════════════════╗
║         🤖 多AI辩论系统 v1.0 🤖          ║
║       Powered by OpenCode Go Models      ║
╚══════════════════════════════════════════╝[/bold bright_yellow]
"""
    console.print(banner)


def print_models(models: list[str]):
    for i, m in enumerate(models, 1):
        info = ALL_MODELS.get(m, {})
        console.print(f"  [bold]{i}.[/bold] {info.get('emoji', '🤖')} {info.get('name', m)} ({m})")


def select_models(default_models: list[str], title: str) -> list[str]:
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("[dim]可用模型:[/dim]")
    all_ids = list(ALL_MODELS.keys())
    print_models(all_ids)
    console.print(f"\n[dim]默认选择: {', '.join(_model_label(m) for m in default_models)}[/dim]")
    choice = Prompt.ask(
        "输入模型编号 (逗号分隔, 回车使用默认)",
        default="",
    )
    if not choice.strip():
        return default_models
    selected = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(all_ids):
                selected.append(all_ids[idx])
    return selected if selected else default_models


def main():
    print_banner()

    while True:
        console.print()
        console.rule("[bold]主菜单[/bold]")
        console.print("  [bold]1.[/bold] 🎙️  圆桌辩论 (多模型轮流发言)")
        console.print("  [bold]2.[/bold] ⚔️  分组对抗辩论 (正方 vs 反方)")
        console.print("  [bold]3.[/bold] 📋 查看可用模型")
        console.print("  [bold]0.[/bold] 🚪 退出")
        console.print()

        mode = Prompt.ask("选择模式", choices=["0", "1", "2", "3"], default="1")

        if mode == "0":
            console.print("[dim]再见！[/dim]")
            break

        if mode == "3":
            console.print()
            table = Table(title="可用模型", show_lines=True)
            table.add_column("模型ID", style="cyan")
            table.add_column("名称", style="bold")
            table.add_column("API类型", style="green")
            from config import OPENAI_MODELS, ANTHROPIC_MODELS
            for mid, info in ALL_MODELS.items():
                api_type = "OpenAI" if mid in OPENAI_MODELS else "Anthropic"
                table.add_row(mid, f"{info['emoji']} {info['name']}", api_type)
            console.print(table)
            continue

        topic = Prompt.ask("\n[bold bright_yellow]请输入辩论话题[/bold bright_yellow]")
        if not topic.strip():
            console.print("[red]话题不能为空[/red]")
            continue

        rounds = IntPrompt.ask("辩论轮次", default=3)

        if mode == "1":
            models = select_models(DEFAULT_DEBATE_MODELS, "选择参与圆桌辩论的模型")
            roundtable_debate(topic, models, rounds)

        elif mode == "2":
            console.print("\n[bold cyan]配置正方[/bold cyan]")
            group_a = select_models(DEFAULT_GROUP_A, "正方模型")
            console.print("\n[bold cyan]配置反方[/bold cyan]")
            group_b = select_models(DEFAULT_GROUP_B, "反方模型")
            group_debate(topic, group_a, group_b, rounds)


if __name__ == "__main__":
    main()
