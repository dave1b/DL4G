"""
Example how to use flask to create a service for one or more players
"""
import sys

sys.path.insert(0, "C:/Users/Dave/Documents/GitHub/DL4G")

import logging
from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber
from intro_basic.set_only_trump_else_random_agent import SetOnlyTrumpElseRandomAgent
from ml.mlp_agent import MLPOnlyTrumpAgent


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:
        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('SetOnlyTrumpElseRandomAgent', SetOnlyTrumpElseRandomAgent())
    app.add_player('AgentRandomSchieber', AgentRandomSchieber())
    app.add_player('MLPOnlyTrumpAgent', MLPOnlyTrumpAgent())
    # app.add_player('random', AgentRandomSchieber())

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8888)
