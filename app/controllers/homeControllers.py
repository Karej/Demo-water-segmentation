from flask_restx import Namespace, Resource, fields
from flask import request, jsonify, session, render_template, make_response, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
import base64
import json

from utility import utils
from pipeline.predict import predict_one
from pipeline.estimate import calculate_water_depth
from config.socket.socket import socketio
import torch


api = Namespace('home', description='home page')

upload_parser = api.parser()
upload_parser.add_argument('device_id', type=int, location='form')
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg']


model_path = os.path.join(os.environ.get('STORAGE'),'weight','best_model.pth')
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.route('/')
class Home(Resource):
    def get(self):
        return make_response(render_template('home.html'))

@api.route('/track')
class Home(Resource):
    def get(self):
        return make_response(render_template('track.html'))

@api.route('/pushImage')
class RunAI(Resource):
    def __init__(self, *args):
        super().__init__(*args)
        self.model = model
        self.device = device
        
    @api.expect(upload_parser, validate=False)
    def post(self):
        args = upload_parser.parse_args()
        file = args['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for uploading'})
            resp.status_code = 400
            return resp
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.environ.get('STORAGE'),'image', filename))
            
            predict_one(filename, self.model, self.device)
            
            resp = jsonify({'message' : 'successfully', 'filename': filename})
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
            resp.status_code = 400
            return resp
        


@api.route('/sendImage')
class HandleMainBackend(Resource):
    def __init__(self, *args):
        super().__init__(*args)
        self.model=model
        self.device = device
        
    def post(self):
        data = request.get_json()
        image_name = f'record_{data["deviceID"]}'
        with open(f'{os.environ.get("STORAGE")}/image/{image_name}.jpg', "wb") as f:
            image = base64.decodebytes(data["image"].encode())
            f.write(image)
            print("Image Received")
        predict_one(f"{image_name}.jpg", self.model, self.device)
        filename_mask = f'{os.environ.get("STORAGE")}/mask/{image_name}.png'
        origin_image = f'{os.environ.get("STORAGE")}/image/{image_name}.jpg'
        
        
        parseJson = json.loads(data["info"])
        
        list_1 =  parseJson[0]
        list_2 =  parseJson[1]
        length_object_1 = parseJson[2] # cm
        length_object_2 = parseJson[3] # cm
        
        level = calculate_water_depth(filename_mask,origin_image,length_object_1, length_object_2, list_1, list_2)
        updateData = {'filename': image_name, 'level': level}
        socketio.emit('currentEvent', updateData)
        return {"success": 1, "level": 1}
               