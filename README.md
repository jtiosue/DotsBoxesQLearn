# DotsBoxesQLearn

- `vanilla_ql.py`. A basic Q-learning implementation for the game Dots and Boxes. I train two models, with each model acting as the enviroment for the other model.

- `deep_ql.py`. A deep Q-learning implementation with PyTorch.



To do: 
- currently, the deep QL works pretty well. But we know that with optimal play, player 1 should always win. The deep QL network doesn't quite get to this. I think part of the problem is that there are times when the output is all zero (see the game_player method). Think of a way to get around this.
- analyze the deep QL loss. Is it overfitting, etc?
- Add in symmetries so that when I build the memory, I also add all the symmetric states.
