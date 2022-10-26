from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

class DeterminationMonteCarloTreeSearchAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.minTrumpValue = 68
        self.round = 0

    def action_trump(self, game_observation: GameObservation) -> int:
       pass

    def action_play_card(self, game_State: GameState) -> int:
        pass


## Use implementation
from jass.arena.arena import Arena
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.agents.agent_random_schieber import AgentRandomSchieber

# Jass Arena for trying Agents
arena = Arena(nr_games_to_play=100, cheating_mode=True)
arena.set_players(InformationSetMonteCarloTreeSearchAgent(), AgentRandomSchieber(), InformationSetMonteCarloTreeSearchAgent(), AgentRandomSchieber())
arena.play_all_games()
# arena.play_game(dealer=NORTH)
print(arena.points_team_0.sum(), arena.points_team_1.sum())
