from . import create_flask, create_dash
from .core.layouts import main_layout_sidebar


# The Flask instance
server = create_flask()

# The Dash instance
app = create_dash(server)


# Push an application context so we can use Flask's 'current_app'
with server.app_context():
    # load the rest of our Dash app
    from . import index  # noqa: F401

    # configure the Dash instance's layout
    app.layout = main_layout_sidebar()
