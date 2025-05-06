
from flask import Flask,request, Response,jsonify
import subprocess

app = Flask(__name__)


@app.route('/score', methods=['POST'])
def score() -> Response:

    try:
        #Make sure to get the data from the request and pass it to your inference file

        data = request.json
        
        if not data:
            return Response("No data provided", status=400)
        
        # Pass the json data to your inference file

        # Replace 'data.py' with your inference filename

        # Check data.py file to see how to access the data 

        command = f'cd src && python3 case_study.py "{data}"'
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        return jsonify({"message": "Inference completed successfully", "output": result.stdout}), 200


    except KeyError:

        raise RuntimeError('Error occured')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)