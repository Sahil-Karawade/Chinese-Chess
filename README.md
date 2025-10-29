# Chinese-Chess
# Chinese Chess AI Engine

An ongoing project to build an AI agent for Chinese Chess (Xiangqi) using Reinforcement Learning and Monte Carlo Tree Search (MCTS). The agent learns through self-play, continuously improving its strategy by playing against itself and learning from the outcomes.

**Live Demo:** [https://d10d1tqpminh4p.cloudfront.net/](https://d10d1tqpminh4p.cloudfront.net/)

**Project Status:** Active development - the AI is currently generating self-play data and training iteratively

---

## Project Overview

This project implements a complete pipeline for training a Chinese Chess AI from scratch:

1. **Game Simulation Framework** - A modular environment for simulating Chinese Chess games
2. **Self-Play Data Generation** - The AI plays against itself to generate training data
3. **Monte Carlo Tree Search (MCTS)** - Guides move selection during self-play
4. **Neural Network Training** - Learns from self-play games to improve policy and value predictions
5. **Deployment** - Full-stack web application deployed on AWS

### What is Chinese Chess (Xiangqi)?

Chinese Chess is a strategic board game similar to Classical Chess but with different pieces and rules. It's played on a 9x10 board with a "river" dividing the two sides. The goal is to checkmate the opponent's General (similar to the King in Classical Chess).

**Key Differences from Classical Chess:**
- Pieces move on intersections, not squares
- Some pieces (like Elephants and Advisors) cannot cross the river
- The Cannon captures by jumping over exactly one piece
- Generals cannot face each other directly across an open file

---

## Key Concepts

### Reinforcement Learning (RL)
A machine learning approach where an agent learns by interacting with an environment and receiving rewards. In this project:
- **Agent:** The Chinese Chess AI
- **Environment:** The game board and rules
- **Actions:** Legal moves
- **Reward:** Win (+1), Loss (-1), Draw (0)

### Monte Carlo Tree Search (MCTS)
A search algorithm that builds a tree of possible game positions by:
1. **Selection:** Navigate the tree using PUCT (Predictor + Upper Confidence bounds applied to Trees), which balances exploitation and exploration guided by neural network priors
2. **Expansion:** Add new leaf node to the tree
3. **Evaluation:** Use the neural network to evaluate the position (returns value estimate and move probabilities)
4. **Backpropagation:** Update all nodes along the path with the evaluation result

This approach, used by AlphaZero, is more efficient than traditional MCTS because it uses neural network evaluation instead of random game simulations.

MCTS combined with neural network guidance is particularly effective for games with large action spaces like Chinese Chess, where exhaustive search is computationally infeasible.

### Implementation Notes

This project follows the AlphaZero approach with some practical enhancements:
- **Core AlphaZero:** Neural network evaluation, PUCT selection, no rollouts
- **Domain knowledge:** Heuristic bonuses for checkmate detection and piece captures to accelerate learning
- **Exploration:** Dirichlet noise at root node (as in original AlphaZero paper)

### Self-Play
The AI improves by playing games against itself:
1. Current AI plays a game using MCTS
2. Game positions and outcomes are stored as training data
3. Neural network is trained on this data
4. Updated AI plays new self-play games
5. Repeat → continuous improvement

### Neural Network Architecture
The neural network takes the board position as input and outputs:
- **Policy:** Probability distribution over legal moves (which move to play)
- **Value:** Estimated win probability from this position (how good is this position)

---

## Architecture

### Backend
- **Language:** Python
- **Deep Learning:** PyTorch
- **Training Infrastructure:** Modal AI (A10 GPU)
- **API Framework:** FastAPI
- **Database:** PostgreSQL (game storage and validation)
- **Data Processing:** Pandas, NumPy, SQLAlchemy

### Frontend
- **Framework:** React
- **Hosting:** AWS CloudFront (CDN)

### Cloud Infrastructure
- **Storage:** AWS S3 (game data, model checkpoints)
- **Compute:** AWS Lambda (serverless functions)
- **Database:** AWS DynamoDB (session data)
- **CDN:** AWS CloudFront (global distribution)
- **Monitoring:** AWS CloudWatch

---

## Current Progress

- ✅ Game simulation framework implemented
- ✅ MCTS algorithm integrated
- ✅ Self-play data generation pipeline working
- ✅ PostgreSQL database for game storage
- ✅ Baseline model trained on Modal AI (A10 GPU)
- ✅ Web application deployed on AWS
- 🔄 **Ongoing:** Generating 30,000+ self-play samples
- 🔄 **Ongoing:** Iterative training and model improvement
- 🔄 **Ongoing:** Hyperparameter tuning and architecture experiments

---

## Features

### Implemented
- Complete Chinese Chess game rules and move validation
- MCTS-guided move selection
- Self-play data generation with configurable parameters
- PostgreSQL integration for data persistence
- Exploratory Data Analysis (EDA) for training data
- Model training pipeline on GPU
- RESTful API for game interaction
- React-based web interface
- AWS deployment with monitoring

### Planned
- [ ] Stronger baseline models with deeper networks
- [ ] Opening book integration
- [ ] Endgame tablebase support
- [ ] Mobile-responsive UI improvements
- [ ] Replay system for analyzing games
- [ ] ELO rating system for tracking improvement

---

## Technologies Used

**Machine Learning & AI:**
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Pandas (data manipulation)

**Backend:**
- FastAPI (REST API)
- PostgreSQL (database)
- SQLAlchemy (ORM)

**Cloud & Infrastructure:**
- Modal AI (GPU training)
- AWS (S3, Lambda, DynamoDB, CloudFront, CloudWatch)

**Frontend:**
- React (UI framework)
- JavaScript/HTML/CSS

**Development Tools:**
- Git (version control)

---

## Learning Resources

If you're new to these concepts, here are some helpful resources:

**Chinese Chess (Xiangqi):**
- [Xiangqi Rules - Wikipedia](https://en.wikipedia.org/wiki/Xiangqi)

**Reinforcement Learning:**
- Sutton & Barto - *Reinforcement Learning: An Introduction*
- [OpenAI Spinning Up](https://spinningup.openai.com/)

**Monte Carlo Tree Search:**
- [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
- [AlphaGo Paper](https://www.nature.com/articles/nature16961) (similar approach)

**Self-Play in Games:**
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) (chess, shogi, go)

---

## Project Goals

1. **Build a strong Chinese Chess AI** that can compete with human players
2. **Learn and implement RL algorithms** in a practical setting
3. **Develop production ML systems** with proper data pipelines and monitoring
4. **Demonstrate end-to-end ML engineering** from research to deployment
5. **Document the journey** for educational purposes

---

## Contributing

This is a personal learning project, but suggestions and feedback are welcome! Feel free to:
- Open issues for bugs or ideas
- Suggest improvements to the architecture
- Share resources about Chinese Chess AI

---

## Author

[Sahil Karawade]
- GitHub: [@Sahil-Karawade](https://github.com/Sahil-Karawade)


---

## Acknowledgments

- Modal AI for GPU infrastructure
- AWS for deployment platform
- The Xiangqi community for game knowledge
- AlphaGo/AlphaZero papers for inspiration

---

**Last Updated:** [October 2025]

**Current Training Status:** Generating self-play data (30,000+ samples collected)