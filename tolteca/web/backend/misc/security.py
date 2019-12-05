from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return _pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return _pwd_context.hash(password)
