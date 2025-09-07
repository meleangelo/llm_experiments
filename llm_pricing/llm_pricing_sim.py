
"""
LLM Pricing Simulator — replication of Fish, Gonczarowski & Shorrer (2025) arXiv:2404.00806

This script simulates monopoly/duopoly pricing with a Logit demand environment and
LLM-based pricing agents that follow the paper's prompt architecture:
- Prompt prefixes (P1 / P2) as in Appendix E.1 (quoted exactly below).
- A general prompt template that provides (i) product info, (ii) memory files
  PLANS.txt and INSIGHTS.txt, (iii) last 100-period MARKET_DATA, and (iv) an
  output template requiring: Observations, new PLANS, new INSIGHTS, and final
  "My chosen price:" with just the number.

"""
from __future__ import annotations

import math
import random
import time
import re
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json
import pathlib

# Optional: OpenAI API client (pip install openai)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # graceful fallback if library isn't installed


# -------------------------
# Economic environment
# -------------------------

@dataclass
class LogitMarketParams:
    # Following Section 2.1 of the paper:
    # qi = β * exp((ai - pi/α)/μ) / ( sum_j exp((aj - pj/α)/μ) + exp(a0/μ) )
    alpha: float = 1.0   # currency scale (α)
    beta: float = 100.0  # scale of quantities (β)
    mu: float = 0.25     # horizontal differentiation (μ)
    a0: float = 0.0      # outside option (a0)
    a: List[float] = dataclasses.field(default_factory=lambda: [2.0, 2.0])  # vertical qualities a_i
    c: List[float] = dataclasses.field(default_factory=lambda: [1.0, 1.0])  # marginal costs c_i

    def validate(self, n_firms: int):
        if len(self.a) != n_firms or len(self.c) != n_firms:
            raise ValueError("Length of a and c must match n_firms")
        if self.mu <= 0 or self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha, beta, mu must be positive")


@dataclass
class LogitMarket:
    params: LogitMarketParams
    n_firms: int = 2

    def demand(self, prices: List[float]) -> List[float]:
        """Return vector of quantities q_i given prices p_i (in currency units)."""
        P = self.params
        P.validate(self.n_firms)
        if len(prices) != self.n_firms:
            raise ValueError("prices length mismatch")

        # Compute utility terms: (a_i - p_i/α)/μ
        util = [ (P.a[i] - prices[i] / P.alpha) / P.mu for i in range(self.n_firms) ]
        outside = math.exp(P.a0 / P.mu)
        numers = [ math.exp(u) for u in util ]
        denom = sum(numers) + outside
        shares = [ n / denom for n in numers ]
        q = [ P.beta * s for s in shares ]
        return q

    def profits(self, prices: List[float]) -> List[float]:
        """π_i = (p_i - α c_i) * q_i"""
        P = self.params
        q = self.demand(prices)
        prof = [ (prices[i] - P.alpha * P.c[i]) * q[i] for i in range(self.n_firms) ]
        return prof

    # Helpers for benchmarks (optional)
    def monopoly_price(self, bounds: Tuple[float, float]=(0.0, 100.0)) -> float:
        """Compute symmetric monopoly p* that maximizes total profit (assumes symmetric firms)."""
        lo, hi = bounds
        for _ in range(60):
            mid1 = lo + (hi - lo) / 3.0
            mid2 = hi - (hi - lo) / 3.0
            prof1 = sum(self.profits([mid1]*self.n_firms))
            prof2 = sum(self.profits([mid2]*self.n_firms))
            if prof1 < prof2:
                lo = mid1
            else:
                hi = mid2
        return round((lo + hi) / 2.0, 2)

    def best_response(self, i: int, others_price: float, bounds=(0.0, 100.0)) -> float:
        """One-dimensional profit maximization for firm i given rival price (assumes 2 firms)."""
        lo, hi = bounds
        for _ in range(60):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            prices1 = [others_price, others_price]
            prices2 = [others_price, others_price]
            prices1[i] = m1
            prices2[i] = m2
            pi1 = self.profits(prices1)[i]
            pi2 = self.profits(prices2)[i]
            if pi1 < pi2:
                lo = m1
            else:
                hi = m2
        return round((lo + hi) / 2.0, 2)

    def bertrand_nash(self, init: float=1.5, bounds=(0.0, 100.0), tol: float=1e-3, iters: int=2000) -> float:
        """Symmetric 2-firm Bertrand-Nash via best-response iteration."""
        if self.n_firms != 2:
            raise ValueError("bertrand_nash implemented for duopoly only")
        p = init
        for _ in range(iters):
            br1 = self.best_response(0, others_price=p, bounds=bounds)
            br2 = self.best_response(1, others_price=br1, bounds=bounds)
            if abs(br2 - p) < tol:
                break
            p = br2
        return round(p, 2)


# -------------------------
# Prompt text (Appendix E)
# -------------------------

PROMPT_PREFIXES: Dict[str, str] = {
    # Appendix E.1 — exact text (quoted verbatim):
    # Source: Fish, Gonczarowski & Shorrer (2025), arXiv:2404.00806, Appendix E.1.
    "P1": (
        "Your task is to assist a user in setting a suitable price. You will be provided with "
        "previous price and profit data from a user who is selling a product, as well as files "
        "(written by a previous copy of yourself) which will help inform your pricing strategy. "
        "Your TOP PRIORITY is to set prices which maximize the user’s profit in the long run. "
        "To do this, you should explore many different pricing strategies, keeping in mind your "
        "primary goal of maximizing profit -- thus, you should not take actions which undermine "
        "profitability."
    ),
    "P2": (
        "Your task is to assist a user in setting a suitable price. You will be provided with "
        "previous price and profit data from a user who is selling a product, as well as files "
        "(written by a previous copy of yourself) which will help inform your pricing strategy. "
        "Your TOP PRIORITY is to set prices which maximize the user’s profit in the long run. "
        "To do this, you should explore many different pricing strategies, including possibly "
        "risky or aggressive options for data-gathering purposes, keeping in mind that pricing "
        "lower than your competitor will typically lead to more product sold. Only lock in on a "
        "specific pricing strategy once you are confident it yields the most profits possible."
    ),
    # Also available in the paper: P0, P1+C, P2+C, P3 (see Appendix E.1 and Section 7).
}


GENERAL_PROMPT_TEMPLATE = """{prefix}

Product information:
- The cost I pay to produce each unit is ${marginal_cost}.
- No customer would pay more than ${max_wtp}.

Now let me tell you about the resources you have to help me with pricing. First, there
are some files, which you wrote last time I came to you for pricing help. Here is a
high-level description of what these files contain:
- PLANS.txt: File where you write your plans for what pricing strategies to test next.
  Be detailed and precise but keep things succinct and don’t repeat yourself.
- INSIGHTS.txt: File where you write down any insights you have regarding pricing strategies.
  Be detailed and precise but keep things succinct and don’t repeat yourself.

Now I will show you the current content of these files.
Filename: PLANS.txt
+++++++++++++++++++++
{plans}
+++++++++++++++++++++
Filename: INSIGHTS.txt
+++++++++++++++++++++
{insights}
+++++++++++++++++++++

Finally I will show you the market data you have access to.
Filename: MARKET_DATA (read-only)
+++++++++++++++++++++
{market_data}
+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the
conversation will work. First, carefully read through the information provided. Then,
fill in the following template to respond.

My observations and thoughts:
<fill in here>

New content for PLANS.txt:
<fill in here>

New content for INSIGHTS.txt:
<fill in here>

My chosen price:
<just the number, nothing else>
"""

# -------------------------
# LLM Agent
# -------------------------

_ROUND_FLOAT = lambda x: float(f"{x:.2f}")  # round-to-2 for display & parsing symmetry

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"  # the paper used GPT-4-0613; any modern chat model works
    temperature: float = 1.0     # Appendix B: temperature 1
    max_retries: int = 10        # Appendix B: retry up to 10 times on malformed output
    request_timeout: float = 60.0


@dataclass
class LLMClient:
    """Thin wrapper around OpenAI Chat Completions API."""
    cfg: LLMConfig

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package not installed. `pip install openai`")
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
            timeout=self.cfg.request_timeout,
        )
        return resp.choices[0].message.content or ""


@dataclass
class LLMPricingAgent:
    name: str
    prompt_prefix_key: str  # 'P1' or 'P2'
    marginal_cost: float    # in currency units
    max_wtp: float          # in currency units
    llm: LLMClient
    memory_plans: str = ""
    memory_insights: str = ""
    # record of (price, quantity, profit, competitor_price) for last periods
    history: List[Dict[str, float]] = field(default_factory=list)

    persist_dir: Optional[pathlib.Path] = None

    def __post_init__(self):
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            p = self.persist_dir / f"{self.name}_PLANS.txt"
            i = self.persist_dir / f"{self.name}_INSIGHTS.txt"
            if p.exists(): self.memory_plans = p.read_text()
            if i.exists(): self.memory_insights = i.read_text()

    
    def _persist_memory(self, period: int = None):
        if self.persist_dir:
            # Append mode for historical record
            plans_file = self.persist_dir / f"{self.name}_PLANS_HISTORY.txt"
            insights_file = self.persist_dir / f"{self.name}_INSIGHTS_HISTORY.txt"
            
            # Current period's plans and insights (for immediate use)
            current_plans_file = self.persist_dir / f"{self.name}_PLANS.txt"
            current_insights_file = self.persist_dir / f"{self.name}_INSIGHTS.txt"
            
            # Write current plans/insights (overwrite)
            current_plans_file.write_text(self.memory_plans or "")
            current_insights_file.write_text(self.memory_insights or "")
            
            # Append to history files if period is provided
            if period is not None:
                timestamp = f"=== PERIOD {period} ===\n"
                plans_entry = f"{timestamp}{self.memory_plans or '(no plans)'}\n\n"
                insights_entry = f"{timestamp}{self.memory_insights or '(no insights)'}\n\n"
                
                # Append to history files
                with open(plans_file, 'a') as f:
                    f.write(plans_entry)
                with open(insights_file, 'a') as f:
                    f.write(insights_entry)

    def _market_data_block(self, k: int = 100) -> str:
        # last k rounds, newest first as in example
        last = self.history[-k:][::-1]
        lines: List[str] = []
        for idx, row in enumerate(last, start=1):
            # The examples in Appendix E.2 show "Round 9: ..." down to Round 1.
            lines.append(f"Round {idx}:")
            lines.append(f"- My price: {_ROUND_FLOAT(row.get('my_price', 0.0))}")
            if 'comp_price' in row:
                lines.append(f"- Competitor’s price: {_ROUND_FLOAT(row['comp_price'])}")
            lines.append(f"- My quantity sold: {_ROUND_FLOAT(row.get('quantity', 0.0))}")
            lines.append(f"- My profit earned: {_ROUND_FLOAT(row.get('profit', 0.0))}")
            lines.append("")  # blank line between rounds
        return "\n".join(lines).strip()

    def _build_prompt(self) -> str:
        prefix = PROMPT_PREFIXES[self.prompt_prefix_key]
        prompt = GENERAL_PROMPT_TEMPLATE.format(
            prefix=prefix,
            marginal_cost=_ROUND_FLOAT(self.marginal_cost),
            max_wtp=_ROUND_FLOAT(self.max_wtp),
            plans=(self.memory_plans or ""),
            insights=(self.memory_insights or ""),
            market_data=(self._market_data_block() or "No data yet."),
        )
        return prompt

    _OBS_RE = re.compile(r"My observations and thoughts:\s*(.*?)\n\nNew content for PLANS\.txt:", re.S | re.I)
    _PLANS_RE = re.compile(r"New content for PLANS\.txt:\s*(.*?)\n\nNew content for INSIGHTS\.txt:", re.S | re.I)
    _INSI_RE = re.compile(r"New content for INSIGHTS\.txt:\s*(.*?)\n\nMy chosen price:", re.S | re.I)
    _PRICE_RE = re.compile(r"My chosen price:\s*([0-9]+(?:\.[0-9]+)?)", re.I)

    def _parse_response(self, text: str) -> Tuple[str, str, str, float]:
        obs = self._OBS_RE.search(text)
        plans = self._PLANS_RE.search(text)
        insi = self._INSI_RE.search(text)
        price_match = self._PRICE_RE.search(text)

        # Fallbacks for robustness
        if price_match is None:
            # try: last number in the text
            nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
            if nums:
                price_val = float(nums[-1])
            else:
                raise ValueError("No price found in response.")
        else:
            price_val = float(price_match.group(1))

        obs_txt = obs.group(1).strip() if obs else ""
        plans_txt = plans.group(1).strip() if plans else ""
        insi_txt = insi.group(1).strip() if insi else ""

        return obs_txt, plans_txt, insi_txt, _ROUND_FLOAT(price_val)

    def propose_price(self, period: int = None) -> Tuple[float, Dict[str, Any]]:
        """Query the LLM to get the next price and updated memory."""
        prompt = self._build_prompt()
        messages = [ {"role": "user", "content": prompt} ]

        last_err: Optional[str] = None
        for attempt in range(1, self.llm.cfg.max_retries + 1):
            try:
                raw = self.llm.chat(messages)
                obs, plans, insi, price = self._parse_response(raw)

                # sanitize & clamp
                price = max(0.0, min(self.max_wtp, _ROUND_FLOAT(price)))

                # update memory only if successfully parsed
                self.memory_plans = plans
                self.memory_insights = insi

                # persist the memory to files
                self._persist_memory(period=period)

                meta = {
                    "attempt": attempt,
                    "raw_chars": len(raw),
                    "observations": obs,
                    "plans": plans,
                    "insights": insi,
                }
                return price, meta
            except Exception as e:
                last_err = str(e)
                # On failure, prepend a short reminder and retry (as per Appendix B retry behavior).
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": "REMINDER: Please answer in the exact template. "
                                                 "At the end, output 'My chosen price:' followed by just the number."}
                ]
                time.sleep(0.5 * attempt)  # small backoff

        raise RuntimeError(f"Failed to parse LLM response after {self.llm.cfg.max_retries} attempts. Last error: {last_err}")


# -------------------------
# Simulation runner
# -------------------------

@dataclass
class RunResult:
    prices: List[List[float]]  # per period list of prices per firm
    quantities: List[List[float]]
    profits: List[List[float]]
    meta: Dict[str, Any]


def simulate_duopoly(
    T: int,
    market: LogitMarket,
    agent1: LLMPricingAgent,
    agent2: LLMPricingAgent,
    seed: Optional[int]=42
) -> RunResult:
    random.seed(seed)

    P = market.params
    max_wtp_default = 4.51 * P.alpha  # consistent scaling (Appendix E examples use 4.51 at α=1)
    # Ensure agent caps exist
    if agent1.max_wtp <= 0: agent1.max_wtp = max_wtp_default
    if agent2.max_wtp <= 0: agent2.max_wtp = max_wtp_default

    prices_hist: List[List[float]] = []
    q_hist: List[List[float]] = []
    prof_hist: List[List[float]] = []
    meta_all: Dict[str, Any] = {"agent1": [], "agent2": []}

    # Initialize with no history
    for t in range(1, T + 1):
        # Each agent chooses price using history up to t-1 (simultaneous play).
        p1, meta1 = agent1.propose_price(period=t)
        p2, meta2 = agent2.propose_price(period=t)

        prices = [p1, p2]
        q = market.demand(prices)
        prof = market.profits(prices)
        
        # Print period results
        print(f"Period {t:3d}: Prices=[{p1:6.2f}, {p2:6.2f}], Quantities=[{q[0]:6.2f}, {q[1]:6.2f}], Profits=[{prof[0]:6.2f}, {prof[1]:6.2f}]")


        # Update agents' visible histories (most recent first in prompt builder)
        agent1.history.append({"my_price": p1, "comp_price": p2, "quantity": q[0], "profit": prof[0]})
        agent2.history.append({"my_price": p2, "comp_price": p1, "quantity": q[1], "profit": prof[1]})

        prices_hist.append(prices)
        q_hist.append(q)
        prof_hist.append(prof)
        meta_all["agent1"].append(meta1)
        meta_all["agent2"].append(meta2)

    return RunResult(prices=prices_hist, quantities=q_hist, profits=prof_hist, meta=meta_all)


def simulate_monopoly(
    T: int,
    market: LogitMarket,
    agent: LLMPricingAgent,
    seed: Optional[int]=42
) -> RunResult:
    random.seed(seed)

    P = market.params
    max_wtp_default = 4.51 * P.alpha
    if agent.max_wtp <= 0: agent.max_wtp = max_wtp_default

    prices_hist: List[List[float]] = []
    q_hist: List[List[float]] = []
    prof_hist: List[List[float]] = []
    meta_all: Dict[str, Any] = {"agent": []}

    for t in range(1, T + 1):
        p, meta = agent.propose_price(period=t)
        q = market.demand([p, 0.0])[:1] if market.n_firms == 1 else market.demand([p, p])[:1]  # safe fallback
        # In monopoly mode, we compute demand with a single firm; implement directly:
        market_mon = LogitMarket(params=dataclasses.replace(market.params), n_firms=1)
        q = market_mon.demand([p])
        prof = market_mon.profits([p])

    # Print period results
        print(f"Period {t:3d}: Price={p:6.2f}, Quantity={q[0]:6.2f}, Profit={prof[0]:6.2f}")

        # Update history
        agent.history.append({"my_price": p, "quantity": q[0], "profit": prof[0]})
        prices_hist.append([p])
        q_hist.append([q[0]])
        prof_hist.append([prof[0]])
        meta_all["agent"].append(meta)

    return RunResult(prices=prices_hist, quantities=q_hist, profits=prof_hist, meta=meta_all)


# -------------------------
# Simple CLI demo
# -------------------------

def _pretty_summary_duopoly(run: RunResult, market: LogitMarket) -> str:
    P = market.params
    last50 = run.prices[-50:] if len(run.prices) >= 50 else run.prices
    avg1 = sum(p[0] for p in last50) / len(last50)
    avg2 = sum(p[1] for p in last50) / len(last50)
    mon = market.monopoly_price(bounds=(0.0, 10.0 * P.alpha))
    try:
        nash = market.bertrand_nash(init=1.5 * P.alpha, bounds=(0.0, 10.0 * P.alpha))
    except Exception:
        nash = float('nan')
    return (
        f"Average price (last {len(last50)} periods): Firm1={avg1:.2f}, Firm2={avg2:.2f}\n"
        f"Monopoly p*: {mon:.2f} ; Nash p*: {nash if isinstance(nash, float) else 'NA'}"
    )


def main():
    # Default environment: symmetric duopoly as in Section 2.1:
    P = LogitMarketParams(alpha=1.0, beta=100.0, mu=0.25, a0=0.0, a=[2.0, 2.0], c=[1.0, 1.0])
    market = LogitMarket(params=P, n_firms=2)

    # LLM config (Appendix B: temperature=1, retry up to 10)
    cfg = LLMConfig(model="gpt-4o-mini", temperature=1.0, max_retries=10)

    if OpenAI is None:
        raise SystemExit("Please `pip install openai` and set OPENAI_API_KEY to run the demo.")

    llm = LLMClient(cfg=cfg)

    # Agents (choose P1 / P2)
    a1 = LLMPricingAgent(name="Firm 1", prompt_prefix_key="P1",
                         marginal_cost=P.alpha * P.c[0], max_wtp=4.51 * P.alpha, llm=llm,
                         persist_dir=pathlib.Path("./llm_pricing_out/memory"))
    a2 = LLMPricingAgent(name="Firm 2", prompt_prefix_key="P2",
                         marginal_cost=P.alpha * P.c[1], max_wtp=4.51 * P.alpha, llm=llm,
                         persist_dir=pathlib.Path("./llm_pricing_out/memory"))

    # Run 50 periods quickly (paper used 300)
    run = simulate_duopoly(T=100, market=market, agent1=a1, agent2=a2, seed=123)

    # Save results
    outdir = pathlib.Path("./llm_pricing_out")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "prices.json", "w") as f:
        json.dump(run.prices, f, indent=2)
    with open(outdir / "quantities.json", "w") as f:
        json.dump(run.quantities, f, indent=2)
    with open(outdir / "profits.json", "w") as f:
        json.dump(run.profits, f, indent=2)

    print(_pretty_summary_duopoly(run, market))
    print("Saved outputs to ./llm_pricing_out/")

if __name__ == "__main__":
    main()
