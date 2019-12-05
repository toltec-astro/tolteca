# Import standard library modules

# Import installed modules
from flask_jwt_extended import JWTManager

# Import app code
from ..db.utils import get_user


def setup_jwt(server, session):

    jwt = JWTManager(server)

    @jwt.user_loader_callback_loader
    def get_current_user(id_):
        return get_user(id_, session)
