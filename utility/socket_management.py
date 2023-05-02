from config.socket.socket import socketio


@socketio.on('connect')
def test_connect(auth):
    print("One user connected !")