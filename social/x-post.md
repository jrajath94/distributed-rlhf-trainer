# X Thread: distributed-rlhf-trainer

## Tweet 1

OpenRLHF has 50,000+ lines of code.

I built a complete RLHF training loop in ~400 lines with the same architecture.

Code: github.com/jrajath94/distributed-rlhf-trainer

Thread on why RLHF doesn't need to be complicated:

## Tweet 2

The problem: RLHF implementations couple everything together.

Reward model + experience collection + PPO optimization + orchestration = one giant class.

When PPO diverges, you're debugging 50k lines.
When you want to swap the reward model, you're refactoring 50k lines.

## Tweet 3

My approach: 4 independent components with one mediator.

- RewardModel: scores (query, response) pairs
- PolicyModel: generates + provides log-probs
- ExperienceCollector: gathers rollouts
- PPOTrainer: clipped PPO + GAE + KL penalty
- RLHFOrchestrator: wires them together

Each fits on one screen.

## Tweet 4

The non-obvious insight: the Experience dataclass is the API boundary.

Once you define what an "experience" is (query, response, log_probs, reward, values), each component only needs to produce or consume that one type.

This is the same pattern that makes microservices work -- shared data contracts.

## Tweet 5

Benchmarks (CPU, batch=8, hidden=128):

- PPO update: 300ms p50, 452ms p99
- Experience collection: 3.9 exp/sec
- Full iteration: 1.85 sec
- 38 tests passing, 86% coverage

Generation is 85% of the cost. The optimization is fast.

## Tweet 6

Star it if this is useful. What should I build next?

github.com/jrajath94/distributed-rlhf-trainer

#AI #MachineLearning #RLHF #PPO #OpenSource #BuildInPublic
