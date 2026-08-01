# Pong AI — Deep Q-Network (DQN)

A Pong-playing AI built from scratch using Deep Q-Network (DQN) reinforcement learning with PyTorch and Pygame. The agent learns to play Pong through self-play against a stochastic opponent, achieving an **80% win rate against a perfect tracking opponent** in pure greedy evaluation after just 300 training episodes.

---

## Demo

The AI (left paddle) trained entirely through reinforcement learning — no hardcoded rules or human demonstrations. It learns purely from reward signals: `+1` for scoring, `-1` for conceding.

---

## Architecture

### State Space (6 inputs)
| Variable | Description |
|----------|-------------|
| `ball.x` | Ball x position |
| `ball.y` | Ball y position |
| `ball.dx` | Ball x velocity |
| `ball.dy` | Ball y velocity |
| `p1.y` | AI paddle position |
| `p2.y` | Opponent paddle position |

### Action Space (3 outputs)
- `UP` — move paddle up
- `STAY` — hold position
- `DOWN` — move paddle down

### Network
```
Input(6) → Linear → ReLU → Linear → Output(3)
              64 neurons
```

### DQN Components
- **Experience Replay** — replay buffer of 100,000 experiences, randomly sampled in batches of 64
- **Target Network** — frozen copy of online network, synced every 50 steps to stabilize training
- **Epsilon-Greedy Exploration** — starts at ε=0.5, decays to ε=0.15 floor over training
- **Gradient Clipping** — prevents exploding gradients during backpropagation
- **Frame Skipping (N=4)** — AI decides once every 4 frames and holds that action, matching DeepMind's Atari DQN approach

---

## Reward Function

| Event | Reward |
|-------|--------|
| Ball scores past opponent | `+1` |
| Ball crosses own paddle x-position | `-1` (once per rally) |
| Ball scored past own side | `0` (already penalized above) |
| Proximity to ball (dense reward) | `0.1 × (-distance/height)` |
| All other frames | `0` |

The dense reward provides per-frame positioning signal, preventing the sparse `+1/-1` from being too infrequent to learn from effectively.

---

## Training

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | `0.0001` (halved to `0.00005` at ep 500) |
| Gamma (discount) | `0.99` |
| Epsilon start | `0.5` |
| Epsilon min | `0.15` |
| Epsilon decay | `0.995` per episode |
| Batch size | `64` |
| Replay buffer | `100,000` |
| Target network sync | Every `50` steps |
| Frame skip | `4` frames per decision |

### Opponent
The AI trains against a **stochastic follower** — 50% of the time it tracks the ball, 50% it makes a random move. This gives the AI a realistic chance to score while still providing meaningful opposition.

### Training Curve
- Episodes 1–240: High exploration (ε decaying from 0.5 → 0.15), agent learns basic ball tracking
- Episodes 240–300: Fixed exploration at ε=0.15, policy solidifies
- Episode 300: Peak performance reached — policy stable, no degradation observed

---

## Results

| Metric | Value |
|--------|-------|
| Pure greedy vs stochastic follower | 20/20 (100%) |
| Pure greedy vs perfect follower | 15-16/20 (75-80%) |
| Avg point differential | ~6.8 at episode 300 |
| Training episodes to peak | 300 |
| Training time | ~30 min (CPU) |

---

## Key Design Decision — Frame Skipping

Frame skipping was the single most impactful change in this project. Without it, the AI made a decision every frame (~7ms at 144fps), where state changes between frames were negligible. This meant the replay buffer filled with nearly identical experiences, making it impossible for Q-values to differentiate between actions.

With frame skipping (N=4), each decision covers meaningful ball movement. The network sees genuinely different states between decisions, making each replay buffer experience far more informative.

| Metric | Without Frame Skip | With Frame Skip |
|--------|-------------------|-----------------|
| Win rate vs stochastic follower | 65% | 100% |
| Win rate vs perfect follower | — | 75-80% |
| Episodes to peak | 800 | 300 |
| Training stability | Degraded after ep700 | Stable throughout |

---

## Project Structure

```
pong-dqn/
├── pong.py          # Pong game with pygame rendering
├── pong_train.py    # Headless training environment (no pygame)
├── pongAI.py        # DQN agent + neural network
├── player.py        # Paddle class
├── ball.py          # Ball class
├── train.py         # Training loop and evaluation
└── saved_models/    # Model checkpoints (.pth files)
```

---

## Setup & Usage

### Install dependencies
```bash
pip install pygame torch
```

### Train the agent
```bash
python train.py
```

### Watch the agent play
```python
# In train.py, call:
visualize()
```

### Load a saved model
```python
import torch
import pongAI

AI1 = pongAI.PongAI()
AI1.online_network.load_state_dict(torch.load('saved_models/BEST_MODEL.pth'))
AI1.update_target_network()
AI1.epsilon = 0  # pure greedy
```

---

## Key Learnings

- **Frame skipping** is critical for continuous action games — single-frame decisions produce nearly identical states, making Q-value differentiation impossible
- **Q-value divergence** is a real problem — solved with small weight initialization, gradient clipping, and low learning rate
- **Reward shaping** is a double-edged sword — dense rewards speed up learning but risk misaligned incentives
- **Catastrophic forgetting** happens when fine-tuning a good model — larger replay buffer and less frequent training mitigates this
- **Tabular Q-learning** is infeasible for continuous state spaces — DQN's neural network generalizes across similar states
- **Credit assignment** in sparse reward environments is handled by gamma — rewards propagate backwards through the Bellman equation over many training steps

---

## References
- [Playing Atari with Deep Reinforcement Learning — Mnih et al., 2013](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Human-level control through deep reinforcement learning — DeepMind, 2015](https://www.nature.com/articles/nature14236)