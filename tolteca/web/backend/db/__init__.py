#! /usr/bin/env python

from .tables import load_tables  # noqa: F401
from .models import load_models  # noqa: F401

from .utils import (
    get_role_by_label,
    create_role,
    get_user_by_email,
    create_user,
    assign_role_to_user,
)


def setup_flask_db(session):

    role = get_role_by_label("default", session)
    if not role:
        role = create_role("default", session)

    TEST_USER = 'test0'

    user = get_user_by_email(TEST_USER, session)
    if not user:
        user = create_user(
            session,
            email=TEST_USER,
            password=TEST_USER,
            is_superuser=True,
        )
        assign_role_to_user(role, user, session)
