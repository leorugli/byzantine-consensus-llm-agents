# Byzantine Consensus Game (BCG)

A multi-agent simulation framework for studying Byzantine fault tolerance using LLM-based agents.

## Overview

This implementation simulates a Byzantine Consensus Game where:
- **Honest agents** try to reach consensus on a common integer value
- **Byzantine agents** attempt to disrupt consensus (identity hidden during simulation)
- All agents use **vLLM** with structured JSON output for decision-making
- Communication happens through the **A2A-sim protocol** (Agent-to-Agent)
- Consensus achieved through **supermajority voting** (≥66% agreement)

## Project Structure

```
byzantine_consensus_game/
├── main.py                    # Single simulation runner
├── run_batch.py               # Batch experiment runner
├── config.py                  # Configuration (models, game params)
├── byzantine_consensus.py     # Game logic and state management
├── bcg_agents.py              # LLM-based agent implementations
├── vllm_agent.py              # vLLM interface with JSON schema
├── a2a_sim.py                 # A2A-sim communication protocol
├── agent_network.py           # Network topology management
├── communication_protocol.py  # Protocol abstractions
├── protocol_factory.py        # Protocol instantiation
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for 14B+ models)
- vLLM 0.6.0+

## Installation

```bash
pip install -r requirements.txt
```

## Models

The framework supports the following models (configured in `config.py`):

| Model | HuggingFace ID | Parameters |
|-------|----------------|------------|
| Qwen3-8B | `Qwen/Qwen3-8B` | 8B |
| Qwen3-14B | `Qwen/Qwen3-14B` | 14B |
| Qwen3-32B | `Qwen/Qwen3-32B` | 32B |
| Mistral-22B | `mistralai/Mistral-Small-Instruct-2409` | 22B |

## Usage

### Single Simulation

```bash
python main.py --honest 4 --byzantine 0 --rounds 15
```

### Batch Experiments

```bash
# List available batch configurations
python run_batch.py --list

# Run a specific batch
python run_batch.py --batch Q1_qwen3-8b_4agents_batch_0 --gpu 0

# Run all batches for a model
python run_batch.py --model qwen3-8b --gpu 0
```

## Configuration

Edit `config.py` to modify:

```python
# Select active model
ACTIVE_MODEL = "qwen3-14b"

# Game parameters
BCG_CONFIG = {
    "num_honest": 4,
    "num_byzantine": 0,
    "value_range": (0, 50),
    "consensus_threshold": 66.0,
    "max_rounds": 15,
}
```

## How It Works

1. **Initialization**: Agents receive random integer values from the configured range
2. **Each Round**:
   - Agents broadcast their current value and reasoning
   - Each agent receives all proposals via A2A protocol
   - LLM generates JSON response with new value and strategy
   - Agents vote to stop (consensus) or continue
3. **Termination**: Game ends when ≥66% vote to stop, or max rounds reached

## Agent Types

- **Honest Agents**: Use LLM to analyze proposals and converge toward consensus
- **Byzantine Agents**: Use LLM to strategically disrupt honest consensus while appearing cooperative

## License

MIT
