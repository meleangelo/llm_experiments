# LLM Pricing Simulator

A Python implementation of the LLM-based pricing simulation from * "Algorithmic Collusion by Large Language Models" by Fish, Gonczarowski & Shorrer (2025)** -  [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806).

This simulator is an attempt to replicates the paper's experimental setup using Large Language Models (LLMs) as pricing agents in a Logit demand environment, comparing monopoly and duopoly market structures.

Feel free to use and modify as you please.

## 📝 License

This project is for my personal research. *Please cite the original paper if you use this code in your research*.


## 🎯 Overview

The simulator implements:
- **Logit demand model** with configurable parameters $$ \alpha, \beta, \mu, a_0, a_i, c_i $$ (see details in the paper) 
- **LLM-based pricing agents** using OpenAI's GPT models
- **Two prompt architectures** (P1 and P2) as specified in the paper's appendices
- **Memory persistence** for agent learning across periods
- **Monopoly and duopoly simulations** with up to 300 periods

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```
   git clone <your-repo-url>
   cd llm_pricing
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

5. **Run the simulation:**
   ```bash
   python llm_pricing_sim.py
   ```

## 📊 What the Simulation Does

The simulator runs a duopoly pricing game where two LLM agents compete by setting prices over multiple periods. Each agent:

1. **Receives market data** from previous periods
2. **Accesses memory files** (PLANS.txt and INSIGHTS.txt) 
3. **Makes pricing decisions** using the specified prompt architecture
4. **Updates memory** based on market outcomes
5. **Learns and adapts** strategies over time

### Default Parameters
- **Market structure:** Symmetric duopoly
- **Demand model:** Logit with $$\alpha=1.0, \beta=100.0, \mu=0.25$$
- **Firm qualities:** $$ a_1=a_2=2.0$$
- **Marginal costs:** $$c_1=c_2=1.0$$
- **Simulation length:** 300 periods (more period, higher cost)

## 🔧 Configuration

### Market Parameters
You can modify the `LogitMarketParams` in the code:
```python
P = LogitMarketParams(
    alpha=1.0,      # Currency scale
    beta=100.0,     # Quantity scale  
    mu=0.25,        # Horizontal differentiation
    a0=0.0,         # Outside option
    a=[2.0, 2.0],   # Vertical qualities
    c=[1.0, 1.0]    # Marginal costs
)
```

### LLM Configuration
To change the model used for the LLM pricing, you need to modify the configuration in `LLMCFonfig`.
Consider also changing the temperature. 
```python
cfg = LLMConfig(
    model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
    temperature=1.0,      # As specified in paper
    max_retries=10        # Retry on malformed output
)
```

### Prompt Architectures
This follows the paper's prompts. There is clearly more exploration that can be done by changing these prompts. 

- **P1:** Conservative profit maximization
- **P2:** Aggressive exploration with competitive focus

## 📈 Output

The simulation generates several output files:

- **Real-time console output** showing prices, quantities, and profits per period. I used this to monitor the simulation over time.

- **JSON files** in `llm_pricing_out/`:
  - `prices.json` - Price history for both firms
  - `quantities.json` - Quantity sold per period
  - `profits.json` - Profit earned per period

- **Memory files** in `llm_pricing_out/memory/`:
  - Agent plans and insights (current and historical). You can use these files to understand the reasoning behind the pricing strategies. It is very interesting to read. In the paper they use some text analysis to summarize these data. I have not implemented that yet. 




## ⚠️ Important Notes

- **API Costs:** Running simulations uses OpenAI API credits. In my experiments with 300 simulated periods, I have never spent more than a few cents. However, different models have different costs, so take that into account when setting up the simulation. The current code runs 1 simulation.

- **Rate Limits:** The code includes delays to respect API rate limits

- **Reproducibility:** Set random seeds for reproducible results

- **Memory Usage:** Long simulations generate large memory files. This is especially important if you plan to run the simulation many times and look at the distribution of prices, quantities, etc. 

