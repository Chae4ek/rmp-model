import pickle
import numpy
import http.server
import urllib.parse

IP = '127.0.0.1'
PORT = 4567

query_key_to_position = {'room_count' : 0,
                         'floor' : 1,
                         'total_floors' : 2,
                         'area' : 3,
                         'kitchen_area' : 4,
                         'living_area' : 5,
                         'ceiling_height' : 6,
                         'repair_type' : 7,
                         'build_year' : 8,
                         'heating_type' : 9,
                         }

rf_model = {}
xgb_model = {}


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_params = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_params.query)
        arr = [0] * 10
        city_id = 0
        for key in query:
            try:
                if key == 'city':
                    city_id = float(query[key][0])
                else:
                    arr[query_key_to_position[key]] = float(query[key][0])
            except (ValueError, OverflowError):
                pass
        flat = numpy.array([arr], dtype="float64")
        rf_predicted = rf_model[city_id].predict(flat)[0]
        xgb_predicted = xgb_model[city_id].predict(flat)[0]
        # response headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        payload = '{{"rf":"{}","xgb":"{}"}}'.format(rf_predicted, xgb_predicted)
        self.wfile.write(payload.encode())
        self.wfile.flush()


if __name__ == '__main__':
    with open("data/models/rf_model_kazan_10features.pkl", "rb") as file:
        rf_model[0] = pickle.load(file)
    with open("data/models/xgboost_model_kazan_10features.pkl", "rb") as file:
        xgb_model[0] = pickle.load(file)
    with open("data/models/rf_model_moscow_10features.pkl", "rb") as file:
        rf_model[1] = pickle.load(file)
    with open("data/models/xgboost_model_moscow_10features.pkl", "rb") as file:
        xgb_model[1] = pickle.load(file)
    with open("data/models/rf_model_omsk_10features.pkl", "rb") as file:
        rf_model[2] = pickle.load(file)
    with open("data/models/xgboost_model_omsk_10features.pkl", "rb") as file:
        xgb_model[2] = pickle.load(file)
    with open("data/models/rf_model_spb_10features.pkl", "rb") as file:
        rf_model[3] = pickle.load(file)
    with open("data/models/xgboost_model_spb_10features.pkl", "rb") as file:
        xgb_model[3] = pickle.load(file)
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        print('Serving on port', PORT)
        httpd.serve_forever()
