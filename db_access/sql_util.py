from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

def get_session(conn_string:str, echo:bool = False)->Session:

    """Return a valid session instance.

    Args:
        conn_string (str): Connection String
        echo (bool, optional): Indicates whether the Session will log the SQL commands os not. Defaults to False.

    Returns:
        Session: SQLAlchemy Session connected to the database.
    """

    engine = create_engine(conn_string, echo=echo)
    Session = sessionmaker(bind=engine)

    return Session()

# Base class that will be inherited to map database tabes to classes
Base = declarative_base()