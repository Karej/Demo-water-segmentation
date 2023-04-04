from flask_restx import Namespace, Resource, fields
from flask import request, jsonify, session, render_template, make_response
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os


from pipeline.predict import predict_one
import torch

api = Namespace('home', description='home page')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg']

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.route('/')
class Home(Resource):
    def get(self):
        return make_response(render_template('home.html'))

@api.route('/pushImage')
class RunAI(Resource):
    def __init__(self, *args):
        super().__init__(*args)
        self.model_path = os.path.join(os.getenv('STORAGE'),'weight','best_model.pth')
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_path, map_location=self.device)
        
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
            file.save(os.path.join(os.getenv('STORAGE'),'image', filename))
            
            predict_one(filename, self.model, self.device)
            
            resp = jsonify({'message' : 'successfully', 'filename': filename})
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
            resp.status_code = 400
            return resp

        