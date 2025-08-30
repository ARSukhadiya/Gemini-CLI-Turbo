"""
Main CLI interface for Gemini CLI with advanced features and performance monitoring.

This module provides:
- Interactive command-line interface
- Streaming responses for real-time interaction
- Performance benchmarking and monitoring
- Configuration management
- Rich output formatting
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns

from .core.client import GeminiClient
from .core.cache import CacheManager
from .core.metrics import PerformanceMonitor
from .config import gemini_config, performance_config, cache_config

# Initialize Typer app
app = typer.Typer(
    name="gemini-cli",
    help="High-Performance Command-Line Interface for Google's Gemini LLM",
    add_completion=False,
    rich_markup_mode="rich"
)

# Rich console for beautiful output
console = Console()

# Global client instance
client: Optional[GeminiClient] = None


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("gemini-cli.log")
        ]
    )


def print_banner():
    """Print the Gemini CLI banner."""
    banner_text = """
[bold blue]╔══════════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║                    🚀 Gemini CLI v1.0.0                    ║[/bold blue]
[bold blue]║              High-Performance AI Interface                 ║[/bold blue]
[bold blue]║                                                          ║[/bold blue]
[bold blue]║  [green]⚡ Ultra-Low Latency[/green] | [yellow]🔄 Async Architecture[/yellow]  ║[/bold blue]
[bold blue]║  [cyan]🧠 Smart Caching[/cyan] | [magenta]📊 Performance Metrics[/magenta]    ║[/bold blue]
[bold blue]╚══════════════════════════════════════════════════════════════╝[/bold blue]
    """
    console.print(banner_text)


def print_status_table():
    """Print current configuration status."""
    table = Table(title="Configuration Status", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    
    # API Configuration
    table.add_row("API Key", "✓ Configured" if gemini_config.api_key else "✗ Not Set", 
                  "✅" if gemini_config.api_key else "❌")
    table.add_row("Model", gemini_config.model, "✅")
    table.add_row("Base URL", gemini_config.api_base_url, "✅")
    
    # Performance Configuration
    table.add_row("Max Concurrent", str(gemini_config.max_concurrent_requests), "✅")
    table.add_row("Timeout", f"{gemini_config.request_timeout}s", "✅")
    table.add_row("Connection Pool", str(gemini_config.connection_pool_size), "✅")
    
    # Cache Configuration
    table.add_row("Cache Enabled", "Yes" if gemini_config.cache_enabled else "No", 
                  "✅" if gemini_config.cache_enabled else "⚠️")
    table.add_row("Cache Strategy", gemini_config.cache_strategy, "✅")
    table.add_row("Cache TTL", f"{gemini_config.cache_ttl}s", "✅")
    
    console.print(table)


async def initialize_client(api_key: Optional[str] = None) -> GeminiClient:
    """Initialize the Gemini client with proper error handling."""
    try:
        if not api_key and not gemini_config.api_key:
            api_key = Prompt.ask(
                "Enter your Gemini API key",
                password=True,
                default=""
            )
            if not api_key:
                console.print("[red]API key is required![/red]")
                sys.exit(1)
        
        client = GeminiClient(api_key or gemini_config.api_key)
        await client.start()
        
        console.print("[green]✓ Gemini client initialized successfully![/green]")
        return client
        
    except Exception as e:
        console.print(f"[red]Failed to initialize Gemini client: {e}[/red]")
        sys.exit(1)


async def interactive_mode():
    """Run the CLI in interactive mode."""
    global client
    
    if not client:
        client = await initialize_client()
    
    console.print("\n[bold cyan]Interactive Mode - Type 'help' for commands, 'quit' to exit[/bold cyan]")
    
    while True:
        try:
            # Get user input
            prompt = Prompt.ask("\n[bold green]Gemini[/bold green]")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            elif prompt.lower() == 'help':
                show_help()
                continue
            elif prompt.lower() == 'status':
                await show_status()
                continue
            elif prompt.lower() == 'stats':
                await show_performance_stats()
                continue
            elif prompt.lower() == 'clear':
                console.clear()
                print_banner()
                continue
            elif not prompt.strip():
                continue
            
            # Process the prompt
            await process_prompt(prompt, stream=True)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit or continue typing...[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def process_prompt(prompt: str, stream: bool = False):
    """Process a single prompt and display the response."""
    global client
    
    if not client:
        console.print("[red]Client not initialized![/red]")
        return
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating response...", total=None)
            
            if stream:
                # Streaming response
                progress.update(task, description="Streaming response...")
                
                response_text = ""
                async for chunk in client.generate_text(prompt, stream=True):
                    response_text += chunk
                    # Update progress with current response length
                    progress.update(task, description=f"Response: {len(response_text)} chars")
                
                # Display final response
                console.print("\n[bold cyan]Response:[/bold cyan]")
                console.print(Panel(response_text, title="Gemini Response", border_style="green"))
                
            else:
                # Synchronous response
                progress.update(task, description="Generating response...")
                response = await client.generate_text(prompt, stream=False)
                
                progress.update(task, description="Response received!")
                time.sleep(0.5)  # Brief pause to show completion
                
                # Display response
                console.print("\n[bold cyan]Response:[/bold cyan]")
                console.print(Panel(response, title="Gemini Response", border_style="green"))
        
        # Cache the response for future use
        if client.cache.is_enabled():
            await client.cache.set(prompt, response_text if stream else response)
        
    except Exception as e:
        console.print(f"[red]Error generating response: {e}[/red]")


def show_help():
    """Show available commands and help information."""
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Basic Commands:[/bold]
  • [green]help[/green] - Show this help message
  • [green]status[/green] - Show current configuration and status
  • [green]stats[/green] - Show performance statistics
  • [green]clear[/green] - Clear the console
  • [green]quit[/green] - Exit the CLI

[bold]Performance Features:[/bold]
  • [yellow]Async Processing[/yellow] - Non-blocking request handling
  • [yellow]Connection Pooling[/yellow] - Optimized HTTP connections
  • [yellow]Smart Caching[/yellow] - Intelligent response caching
  • [yellow]Circuit Breaker[/yellow] - Fault tolerance protection
  • [yellow]Real-time Metrics[/yellow] - Performance monitoring

[bold]Usage Tips:[/bold]
  • Type your questions naturally
  • Use 'quit' to exit gracefully
  • Check 'stats' for performance insights
  • Responses are automatically cached for speed
    """
    
    console.print(Panel(help_text, title="Help", border_style="blue"))


async def show_status():
    """Show current system status and configuration."""
    global client
    
    if not client:
        console.print("[red]Client not initialized![/red]")
        return
    
    # Get health status
    health = client.get_health_status()
    
    # Create status table
    table = Table(title="System Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Client status
    table.add_row("Client", health["status"], 
                  f"Active connections: {health['active_connections']}")
    
    # Circuit breaker
    cb_status = health["circuit_breaker"]
    cb_color = "green" if cb_status == "CLOSED" else "red" if cb_status == "OPEN" else "yellow"
    table.add_row("Circuit Breaker", f"[{cb_color}]{cb_status}[/{cb_color}]", 
                  f"Pool size: {health['connection_pool_size']}")
    
    # Cache status
    cache_stats = await client.cache.get_stats()
    cache_status = "Enabled" if client.cache.is_enabled() else "Disabled"
    table.add_row("Cache", cache_status, 
                  f"Hit rate: {cache_stats.get('performance', {}).get('hit_rate', 0):.1%}")
    
    console.print(table)


async def show_performance_stats():
    """Show comprehensive performance statistics."""
    global client
    
    if not client:
        console.print("[red]Client not initialized![/red]")
        return
    
    try:
        stats = await client.get_performance_stats()
        
        # Create performance overview
        overview = Table(title="Performance Overview", show_header=True, header_style="bold magenta")
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")
        overview.add_column("Status", style="yellow")
        
        # Request statistics
        requests = stats["requests"]
        overview.add_row("Total Requests", str(requests["total"]), "📊")
        overview.add_row("Cache Hit Rate", f"{requests['cache_hit_rate']:.1%}", 
                        "✅" if requests['cache_hit_rate'] > 0.5 else "⚠️")
        overview.add_row("Active Requests", str(requests["active"]), "🔄")
        
        # Circuit breaker
        cb = stats["circuit_breaker"]
        cb_color = "green" if cb["state"] == "CLOSED" else "red" if cb["state"] == "OPEN" else "yellow"
        overview.add_row("Circuit Breaker", f"[{cb_color}]{cb['state']}[/{cb_color}]", 
                        f"Failures: {cb['failure_count']}")
        
        console.print(overview)
        
        # Detailed metrics
        if "metrics" in stats and stats["metrics"]:
            metrics = stats["metrics"]
            
            if "latency" in metrics and metrics["latency"]:
                latency_table = Table(title="Latency Distribution", show_header=True, header_style="bold blue")
                latency_table.add_column("Percentile", style="cyan")
                latency_table.add_column("Latency (ms)", style="green")
                
                percentiles = metrics["latency"].get("percentiles", {})
                for p, value in percentiles.items():
                    latency_table.add_row(p.upper(), f"{value:.1f}")
                
                console.print(latency_table)
        
    except Exception as e:
        console.print(f"[red]Failed to get performance stats: {e}[/red]")


async def benchmark_mode(iterations: int = 10, prompts: Optional[List[str]] = None):
    """Run performance benchmarking."""
    global client
    
    if not client:
        client = await initialize_client()
    
    if not prompts:
        prompts = [
            "Explain quantum computing in simple terms",
            "Write a Python function for binary search",
            "What are the benefits of renewable energy?",
            "Explain machine learning to a beginner",
            "Write a haiku about technology"
        ]
    
    console.print(f"[bold cyan]Running benchmark with {iterations} iterations...[/bold cyan]")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Benchmarking...", total=iterations * len(prompts))
        
        for i in range(iterations):
            for j, prompt in enumerate(prompts):
                start_time = time.time()
                
                try:
                    response = await client.generate_text(prompt, use_cache=False)
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    
                    results.append({
                        "iteration": i + 1,
                        "prompt": j + 1,
                        "latency": latency,
                        "success": True,
                        "response_length": len(response)
                    })
                    
                except Exception as e:
                    latency = (time.time() - start_time) * 1000
                    results.append({
                        "iteration": i + 1,
                        "prompt": j + 1,
                        "latency": latency,
                        "success": False,
                        "error": str(e)
                    })
                
                progress.advance(task)
    
    # Analyze results
    await analyze_benchmark_results(results)


async def analyze_benchmark_results(results: List[Dict[str, Any]]):
    """Analyze and display benchmark results."""
    if not results:
        console.print("[red]No benchmark results to analyze![/red]")
        return
    
    # Calculate statistics
    latencies = [r["latency"] for r in results if r["success"]]
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        # Display results
        results_table = Table(title="Benchmark Results", show_header=True, header_style="bold green")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Status", style="yellow")
        
        results_table.add_row("Total Requests", str(total_count), "📊")
        results_table.add_row("Successful", str(success_count), "✅")
        results_table.add_row("Success Rate", f"{(success_count/total_count)*100:.1f}%", 
                            "✅" if success_count/total_count > 0.9 else "⚠️")
        results_table.add_row("Average Latency", f"{avg_latency:.1f}ms", 
                            "✅" if avg_latency < 1000 else "⚠️")
        results_table.add_row("Min Latency", f"{min_latency:.1f}ms", "🚀")
        results_table.add_row("Max Latency", f"{max_latency:.1f}ms", "📈")
        results_table.add_row("P50 Latency", f"{p50:.1f}ms", "📊")
        results_table.add_row("P90 Latency", f"{p90:.1f}ms", "📊")
        results_table.add_row("P99 Latency", f"{p99:.1f}ms", "📊")
        
        console.print(results_table)
        
        # Performance insights
        insights = []
        if avg_latency < 500:
            insights.append("🎯 Excellent performance! Latency is very low.")
        elif avg_latency < 1000:
            insights.append("✅ Good performance! Latency is acceptable.")
        else:
            insights.append("⚠️ Performance could be improved. Consider optimization.")
        
        if success_count/total_count > 0.95:
            insights.append("🔒 High reliability! Very few failures.")
        else:
            insights.append("⚠️ Some reliability issues detected.")
        
        if insights:
            console.print("\n[bold cyan]Performance Insights:[/bold cyan]")
            for insight in insights:
                console.print(f"  {insight}")
    
    else:
        console.print("[red]No successful requests to analyze![/red]")


@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="Prompt to send to Gemini"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Enable streaming responses"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode"),
    benchmark: bool = typer.Option(False, "--benchmark", "-b", help="Run performance benchmark"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of benchmark iterations"),
    status: bool = typer.Option(False, "--status", help="Show system status"),
    stats: bool = typer.Option(False, "--stats", help="Show performance statistics"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version information")
):
    """High-Performance Command-Line Interface for Google's Gemini LLM."""
    
    # Show version
    if version:
        console.print(f"[bold blue]Gemini CLI v1.0.0[/bold blue]")
        return
    
    # Setup logging
    setup_logging(log_level)
    
    # Print banner
    print_banner()
    
    # Show configuration status
    print_status_table()
    
    # Main logic
    if status:
        asyncio.run(show_status())
    elif stats:
        asyncio.run(show_performance_stats())
    elif benchmark:
        asyncio.run(benchmark_mode(iterations))
    elif interactive:
        asyncio.run(interactive_mode())
    elif prompt:
        asyncio.run(process_single_prompt(prompt, stream))
    else:
        # Default to interactive mode
        asyncio.run(interactive_mode())


async def process_single_prompt(prompt: str, stream: bool):
    """Process a single prompt in non-interactive mode."""
    global client
    
    client = await initialize_client()
    
    try:
        await process_prompt(prompt, stream)
    finally:
        if client:
            await client.close()


if __name__ == "__main__":
    app()

