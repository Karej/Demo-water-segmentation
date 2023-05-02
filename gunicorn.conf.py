import os
from dotenv import load_dotenv
import multiprocessing

for env_file in ('.env', '.flaskenv'):
    env = os.path.join(os.getcwd(), env_file)
    if os.path.exists(env):
        load_dotenv(env)
bind = "127.0.0.1:"+os.environ.get("FLASK_RUN_PORT")
workers = 1
timeout=0