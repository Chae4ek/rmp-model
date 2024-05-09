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


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_params = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_params.query)
        arr = [0] * 10
        for key in query:
            try:
                arr[query_key_to_position[key]] = float(query[key][0])
            except (ValueError, OverflowError):
                pass
        flat = numpy.array([arr], dtype="float64")
        rf_predicted = rf_model.predict(flat)[0]
        xgb_predicted = xgb_model.predict(flat)[0]
        # response headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        payload = '{{"rf":"{}","xgb":"{}"}}'.format(rf_predicted, xgb_predicted)
        self.wfile.write(payload.encode())
        self.wfile.flush()


if __name__ == '__main__':
    with open("data/models/rf_model.pkl", "rb") as file:
        rf_model = pickle.load(file)
    with open("data/models/xgboost_model.pkl", "rb") as file:
        xgb_model = pickle.load(file)
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        print('Serving on port', PORT)
        httpd.serve_forever()
