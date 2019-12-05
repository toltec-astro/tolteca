from .models import User
from .models import Role
from ..misc.security import get_password_hash


def get_user_by_email(email, session):
    return session.query(User).filter(User.email == email).first()


def get_user_by_id(id_, session):
    user = session.query(User).filter(User.pk == id_).first()
    return user


def get_user_hashed_password(user):
    return user.password


def get_user_id(user):
    return user.id


def check_if_user_is_superuser(user):
    return user.is_superuser


def check_if_user_is_active(arg, *args):
    if isinstance(arg, User):
        user = arg
    else:
        user = get_user_by_email(arg, *args)
    return user.is_active


def get_users(session):
    return session.query(User).all()


def get_role_by_label(label, session):
    role = session.query(Role).filter(Role.label == label).first()
    return role


def get_role_by_id(id_, session):
    role = session.query(Role).filter(Role.id == id_).first()
    return role


def get_roles(session):
    return session.query(Role).all()


def get_user_roles(user):
    return user.roles


def create_user(
        session, email, password, **kwargs):
    user = User(
        email=email,
        password=get_password_hash(password),
        **kwargs
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def create_role(label, session):
    role = Role(label=label)
    session.add(role)
    session.commit()
    session.refresh(role)
    return role


def assign_role_to_user(role: Role, user: User, session):
    user.roles.append(role)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
