
from jass.arena.arena import Arena
from jass.game.game_util import *
from jass.game.const import *
from jass.agents.agent_random_schieber import AgentRandomSchieber

from intro_basic.set_only_trump_else_random_agent import SetOnlyTrumpElseRandomAgent
#

# Jass Arena for trying Agents
arena = Arena(nr_games_to_play=100)
arena.set_players(SetOnlyTrumpElseRandomAgent(), AgentRandomSchieber(), SetOnlyTrumpElseRandomAgent(), AgentRandomSchieber())
arena.play_all_games()
# arena.play_game(dealer=NORTH)
print(arena.points_team_0.sum(), arena.points_team_1.sum())
