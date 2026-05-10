"""
LLM Cost Optimizer — Load Test Simulator
Sends concurrent queries in waves to demonstrate batching, caching, and model selection.

Usage:
    python load_test.py                  # Run all 3 waves
    python load_test.py --waves 1        # Run only wave 1
    python load_test.py --waves 1 2      # Run waves 1 and 2
    python load_test.py --delay 2        # 2 second pause between waves
"""
import json
import time
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "http://localhost:8000"
PROMPTS_FILE = Path(__file__).parent / "prompts.json"

def _set_base_url(url: str):
    global BASE_URL
    BASE_URL = url

# Colors for terminal output
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


# =============================================================================
# Core Functions
# =============================================================================
def send_query(prompt_data: dict, index: int) -> dict:
    """Send a single query to the API and return results"""
    query = prompt_data["query"]
    start = time.time()

    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": query, "max_tokens": 500, "temperature": 0.7},
            timeout=120
        )
        elapsed = (time.time() - start) * 1000

        if r.status_code == 200:
            data = r.json()
            return {
                "index": index,
                "query": query[:60],
                "intent": prompt_data.get("intent", "—"),
                "cached": data.get("cached", False),
                "model": data.get("selected_model", "—"),
                "tokens_used": data.get("tokens_used", 0),
                "tokens_saved": data.get("tokens_saved", 0),
                "cost": data.get("cost", 0),
                "cost_saved": data.get("cost_saved", 0),
                "latency_ms": data.get("latency_ms", elapsed),
                "batch_id": data.get("batch_id", "—"),
                "expected": prompt_data.get("expect", "cache_miss"),
                "success": True,
            }
        else:
            return {"index": index, "query": query[:60], "success": False, "error": f"HTTP {r.status_code}"}

    except Exception as e:
        return {"index": index, "query": query[:60], "success": False, "error": str(e)}


def run_wave(wave_name: str, wave_data: dict, concurrency: int = 8):
    """Run a single wave of concurrent queries"""
    label = wave_data["label"]
    prompts = wave_data["prompts"]
    count = len(prompts)

    print(f"\n{C.BOLD}{C.BLUE}{'━' * 70}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  {wave_name.upper()}: {label}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  Sending {count} queries concurrently...{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'━' * 70}{C.END}\n")

    results = []
    wave_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_query, p, i): i
            for i, p in enumerate(prompts, 1)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.get("success"):
                cached = result["cached"]
                icon = f"{C.GREEN}✓ HIT {C.END}" if cached else f"{C.YELLOW}✗ MISS{C.END}"
                model = result['model'] or '—'

                print(
                    f"  {icon}  "
                    f"{C.DIM}#{result['index']:02d}{C.END}  "
                    f"{result['query']:<55s}  "
                    f"{C.CYAN}{model:<18s}{C.END}  "
                    f"${result['cost']:.6f}  "
                    f"{result['latency_ms']:>7.0f}ms"
                )
            else:
                print(f"  {C.RED}✗ FAIL  #{result['index']:02d}  {result.get('error', 'Unknown')}{C.END}")

    wave_time = (time.time() - wave_start) * 1000

    # Wave summary
    successful = [r for r in results if r.get("success")]
    hits = sum(1 for r in successful if r["cached"])
    misses = len(successful) - hits
    total_cost = sum(r["cost"] for r in successful)
    total_saved = sum(r["cost_saved"] for r in successful)
    avg_latency = sum(r["latency_ms"] for r in successful) / max(len(successful), 1)

    models_used = {}
    batches = set()
    for r in successful:
        if r["model"] and r["model"] != "—":
            models_used[r["model"]] = models_used.get(r["model"], 0) + 1
        if r["batch_id"] != "—" and r["batch_id"] is not None:
            batches.add(r["batch_id"])

    print(f"\n  {C.DIM}{'─' * 66}{C.END}")
    print(f"  {C.BOLD}Wave Summary:{C.END}  "
          f"{C.GREEN}{hits} hits{C.END} · "
          f"{C.YELLOW}{misses} misses{C.END} · "
          f"Cost: ${total_cost:.6f} · "
          f"Saved: ${total_saved:.6f} · "
          f"Avg: {avg_latency:.0f}ms · "
          f"Total: {wave_time:.0f}ms")

    if models_used:
        models_str = ", ".join(f"{m}({c})" for m, c in sorted(models_used.items(), key=lambda x: -x[1]))
        print(f"  {C.BOLD}Models:{C.END}      {models_str}")

    if batches:
        print(f"  {C.BOLD}Batches:{C.END}     {len(batches)} created ({', '.join(sorted(batches))})")

    return results


def print_final_report(all_results: list):
    """Print the final summary report"""
    successful = [r for r in all_results if r.get("success")]
    failed = [r for r in all_results if not r.get("success")]

    total = len(all_results)
    hits = sum(1 for r in successful if r["cached"])
    misses = len(successful) - hits
    total_cost = sum(r["cost"] for r in successful)
    total_saved = sum(r["cost_saved"] for r in successful)
    tokens_used = sum(r["tokens_used"] for r in successful)
    tokens_saved = sum(r["tokens_saved"] for r in successful)

    # Cost reduction percentage
    potential_cost = total_cost + total_saved
    reduction_pct = (total_saved / potential_cost * 100) if potential_cost > 0 else 0

    # Model distribution
    models = {}
    for r in successful:
        if r["model"] and r["model"] != "—":
            models[r["model"]] = models.get(r["model"], 0) + 1

    # Intent distribution
    intents = {}
    for r in successful:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1

    print(f"\n\n{C.BOLD}{'━' * 70}{C.END}")
    print(f"{C.BOLD}  LOAD TEST REPORT{C.END}")
    print(f"{C.BOLD}{'━' * 70}{C.END}\n")

    print(f"  {'Queries Sent':<25s} {total}")
    print(f"  {'Successful':<25s} {len(successful)}")
    if failed:
        print(f"  {C.RED}{'Failed':<25s} {len(failed)}{C.END}")
    print()

    print(f"  {C.BOLD}Cache Performance{C.END}")
    print(f"  {'├─ Hits':<25s} {C.GREEN}{hits}{C.END}")
    print(f"  {'├─ Misses':<25s} {C.YELLOW}{misses}{C.END}")
    hit_rate = (hits / max(total, 1)) * 100
    print(f"  {'└─ Hit Rate':<25s} {hit_rate:.1f}%")
    print()

    print(f"  {C.BOLD}Cost Analysis{C.END}")
    print(f"  {'├─ Total Cost':<25s} ${total_cost:.6f}")
    print(f"  {'├─ Cost Saved':<25s} ${total_saved:.6f}")
    print(f"  {'└─ Cost Reduction':<25s} {reduction_pct:.1f}%")
    print()

    print(f"  {C.BOLD}Token Usage{C.END}")
    print(f"  {'├─ Tokens Used':<25s} {tokens_used:,}")
    print(f"  {'└─ Tokens Saved':<25s} {tokens_saved:,}")
    print()

    if models:
        print(f"  {C.BOLD}Model Selection{C.END}")
        for i, (model, count) in enumerate(sorted(models.items(), key=lambda x: -x[1])):
            prefix = "└─" if i == len(models) - 1 else "├─"
            bar = "█" * min(count, 20)
            print(f"  {prefix} {model:<22s} {bar} ({count})")
        print()

    if intents:
        print(f"  {C.BOLD}Intent Coverage{C.END}")
        for i, (intent, count) in enumerate(sorted(intents.items(), key=lambda x: -x[1])):
            prefix = "└─" if i == len(intents) - 1 else "├─"
            print(f"  {prefix} {intent:<22s} {count} queries")
        print()

    print(f"{'━' * 70}")
    print(f"  Dashboard: {C.CYAN}http://localhost:8501{C.END}")
    print(f"  API Docs:  {C.CYAN}http://localhost:8000/docs{C.END}")
    print(f"{'━' * 70}\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="LLM Cost Optimizer — Load Test")
    parser.add_argument("--waves", nargs="+", type=int, default=[1, 2, 3],
                        help="Which waves to run (1, 2, 3). Default: all")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to pause between waves. Default: 1.0")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent requests per wave. Default: 3")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="Backend URL. Default: http://localhost:8000")
    args = parser.parse_args()

    # Update module-level URL
    _set_base_url(args.url)

    # Check backend
    print(f"\n{C.BOLD}⚡ LLM Cost Optimizer — Load Test{C.END}")
    print(f"{C.DIM}   Backend: {BASE_URL}{C.END}")

    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        status = r.json()
        sim_llm = status.get("simulation_mode", {}).get("llm", "?")
        sim_emb = status.get("simulation_mode", {}).get("embeddings", "?")
        print(f"{C.DIM}   Status:  Online (LLM sim={sim_llm}, Embeddings sim={sim_emb}){C.END}")
    except Exception:
        print(f"\n{C.RED}  ✗ Cannot connect to {BASE_URL}{C.END}")
        print(f"{C.DIM}    Start the backend: python -m uvicorn main:app --port 8000{C.END}\n")
        sys.exit(1)

    # Load prompts
    if not PROMPTS_FILE.exists():
        print(f"\n{C.RED}  ✗ Prompts file not found: {PROMPTS_FILE}{C.END}\n")
        sys.exit(1)

    with open(PROMPTS_FILE, "r") as f:
        prompts_data = json.load(f)

    wave_map = {
        1: ("Wave 1", prompts_data.get("wave_1", {})),
        2: ("Wave 2", prompts_data.get("wave_2", {})),
        3: ("Wave 3", prompts_data.get("wave_3", {})),
    }

    # Run waves
    all_results = []

    for wave_num in args.waves:
        if wave_num not in wave_map:
            print(f"{C.YELLOW}  Skipping unknown wave {wave_num}{C.END}")
            continue

        wave_name, wave_data = wave_map[wave_num]
        if not wave_data or not wave_data.get("prompts"):
            print(f"{C.YELLOW}  Skipping empty {wave_name}{C.END}")
            continue

        results = run_wave(wave_name, wave_data, concurrency=args.concurrency)
        all_results.extend(results)

        # Pause between waves
        if wave_num != args.waves[-1]:
            print(f"\n{C.DIM}  Pausing {args.delay}s before next wave...{C.END}")
            time.sleep(args.delay)

    # Final report
    if all_results:
        print_final_report(all_results)


if __name__ == "__main__":
    main()
