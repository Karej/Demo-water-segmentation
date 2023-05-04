#!/bin/sh
export PYTHONPATH=$PWD
gunicorn wsgi:app