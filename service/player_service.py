import sys

sys.path.insert(0, "C:/Users/Dave/Documents/GitHub/DL4G")
# sys.path.insert(0, "/home/localadmin/DL4G")

import logging
from jass.service.player_service_app import PlayerServiceApp
from determination_monte_carlo_tree_search.dmcts_agent import DMCTSAgent


def create_app():
    logging.basicConfig(level=logging.DEBUG)
    app = PlayerServiceApp('player_service')

    # add some players
    # app.add_player('SetOnlyTrumpElseRandomAgent', SetOnlyTrumpElseRandomAgent())
    # app.add_player('AgentRandomSchieber', AgentRandomSchieber())
    # app.add_player('MLPOnlyTrumpAgent', MLPOnlyTrumpAgent())
    app.add_player('DMCTS_MLP', DMCTSAgent())
    # app.add_player('random', AgentRandomSchieber())

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8080)
