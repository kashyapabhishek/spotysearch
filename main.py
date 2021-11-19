from flask import Flask
from flask_restful import reqparse, Resource, Api
from model_builder.model_builder import ModelBuilder
from model_builder.model_builder import ModelBuilder

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('query')
model_builder_obj = ModelBuilder()

class Train(Resource):

    def get(self):
        obj = ModelBuilder()
        obj.build_model()
        return "completed!"

class Search(Resource):

    def __init__(self):
        super().__init__()
        

    def post(self):
        args = parser.parse_args()
        data = args['query']
        if data is not None:
            return model_builder_obj.search([data])
        return 'requreq a query'


api.add_resource(Search, '/search')
api.add_resource(Train, 'train')



if __name__ == '__main__':
    app.run(debug=True)