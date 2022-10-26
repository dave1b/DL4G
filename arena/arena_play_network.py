import logging

from jass.agents.agent_by_network import AgentByNetwork
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from ml.mlp_agent import MLPOnlyTrumpAgent


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.INFO)

    # setup the arena
    arena = Arena(nr_games_to_play=50)
    player = AgentRandomSchieber()
    my_player = AgentByNetwork('http://178.196.170.109:8888/MLPOnlyTrumpAgent')
    my_player2 = AgentByNetwork('http://178.196.170.109:8888/AgentRandomSchieber')
    # my_player = AgentByNetwork('https://lg3bsb3a96.execute-api.eu-central-1.amazonaws.com/dev/random')

    arena.set_players(my_player, my_player2, my_player, my_player2)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
