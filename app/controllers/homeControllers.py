from flask_restx import Namespace, Resource, fields
from flask import request, jsonify, session, render_template, make_response, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
import base64
from utility import utils


from pipeline.predict import predict_one
from pipeline.estimate import calculate_water_depth
import torch

api = Namespace('home', description='home page')

upload_parser = api.parser()
upload_parser.add_argument('device_id', type=int, location='form')
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg']


model_path = os.path.join(os.getenv('STORAGE'),'weight','best_model.pth')
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)


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
            
            
            # list_1 = os.listdir(os.path.join(os.environ.get('STORAGE'),'list1'))  # này Đồng tự thiết kế database để lưu các điểm nha
            # list_2 = os.listdir(os.path.join(os.environ.get('STORAGE'),'list2'))
            # length_object_1 = os.path.join(os.environ.get('STORAGE'),'length_object_1') # cm
            # length_object_2 = os.path.join(os.environ.get('STORAGE'),'length_object_2') # cm
            
            
            # đây là mẫu cho ảnh 1
            list_1 = [[118, 523], [119, 632], [387, 510], [391, 617], [451, 506], [452, 613], [1140, 470], [1140, 550], [1201, 469], [1201, 551], [1262, 466], [1261, 549], [59, 526], [58, 636]]
            list_2 = [[391, 333], [393, 506], [512, 335], [511, 502], [90, 322], [90, 522], [1230, 343], [1232, 467]]
            length_object_1 = 50 # cm
            length_object_2 = 200 # cm
            
            # đây là mẫu cho ảnh 2
            # list_1 = [[61, 143], [59, 210], [122, 145], [120, 209], [181, 143], [181, 209], [661, 141], [660, 210], [721, 143], [721, 216], [750, 141], [751, 219], [780, 140], [781, 215]]
            # list_2 = [[570, 29], [572, 98], [601, 28], [600, 98], [631, 26], [631, 97], [541, 29], [541, 101]]
            # length_object_1 = 50 # cm
            # length_object_2 = 200 # cm
            ###########################
            
            # image = utils.load_image_in_PIL(os.path.join(os.getenv('STORAGE'),'image',filename))
            # mask_size = (image.size[1],image.size[0])
            mask_size = (1552,809)  # cái này để resize lại mask
            
            
            level = calculate_water_depth(filename_mask,mask_size,length_object_1, length_object_2, list_1, list_2)
            
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
        image_name = f'record_{data["deviceID"]}.jpg'
        with open(f'{os.environ.get("STORAGE")}/image/{image_name}', "wb") as f:
            image = base64.decodebytes(data["image"].encode())
            f.write(image)
            print("Image Received")
        predict_one(image_name, self.model, self.device)
        return {"msg": "done"}
               