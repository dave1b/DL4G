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
    my_player = AgentByNetwork('http://178.196.170.109:8080/MLPOnlyTrumpAgent')
    my_player2 = AgentByNetwork('http://178.196.170.109:8080/AgentRandomSchieber')
    mlp = AgentByNetwork('http://dl4g-h22-dbrunner.enterpriselab.ch:8080/DMCTS_MLP')
    nico = AgentByNetwork('http://147.88.62.113:8080/dmtcs_player_nw')
    onlyTrump = AgentByNetwork('http://dl4g-h22-dbrunner.enterpriselab.ch:8080/OnlyTrump')
    # my_player = AgentByNetwork('https://lg3bsb3a96.execute-api.eu-central-1.amazonaws.com/dev/random')

    arena.set_players(mlp, nico, mlp, nico)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
