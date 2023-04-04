from flask import Flask, render_template, send_from_directory, make_response, redirect
from flask_cors import CORS

import os
from datetime import timedelta

from app.controllers import api


app = Flask(__name__, static_url_path='', static_folder='storage',template_folder='resources/view')
CORS(app)

app.config['PROPAGATE_EXCEPTIONS'] = True




# app.config["JWT_SECRET_KEY"]=os.environ.get('JWT_SECRET_KEY')
# app.config['JWT_TOKEN_LOCATION'] = ['cookies']
# app.config['JWT_COOKIE_CSRF_PROTECT'] = True
# app.config['JWT_ACCESS_COOKIE_PATH'] = '/api/'
# app.config['JWT_REFRESH_COOKIE_PATH'] = '/api/authenticate/refresh'
# app.config['JWT_ACCESS_CSRF_COOKIE_PATH'] = '/api/'
# app.config['JWT_REFRESH_CSRF_COOKIE_PATH'] = '/api/authenticate/refresh'
# app.config['JWT_COOKIE_SECURE'] = False
# app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(minutes=1)
# app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(minutes=90)

# @app.errorhandler(404)
# def index(e):
#     return  make_response(render_template('index.html'))
api.init_app(app)

@app.errorhandler(404)
def index(e):
    return  make_response(redirect('/home/'))

# JWT
