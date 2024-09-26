# DotsBoxesQLearn

- `vanilla_ql.py`. A basic Q-learning implementation for the game Dots and Boxes. Note that player 1 can always force a win. After playing one million games, the model never loses as player 1. If player 1 behaves somewhat nonoptimally, then the model wins or ties a lot as player 2.
- `deep_ql.py`. A deep Q-learning implementation with PyTorch.



To do: 
- currently, the deep QL works pretty well. But we know that with optimal play, player 1 should always win. The deep QL network doesn't quite get to this. I think part of the problem is that there are times when the output is all zero (see the game_player method). Think of a way to get around this.
- analyze the deep QL loss. Is it overfitting, etc?
- Add in symmetries so that when I build the memory, I also add all the symmetric states.
- Right now, with one million games, vanilla Q learning becomes optimal for a 2 by 2 grid. For 3 by 3, presumably it will not be able to get close to optimal. Can we see deep qlearning outperform vanilla q learning for 3 by 3?
