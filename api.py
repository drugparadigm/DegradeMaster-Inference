from flask import Flask,request, Response,jsonify
import subprocess
from datetime import datetime
import os
import traceback
import shutil
import json
from src.case_study import main
app = Flask(__name__)

UPLOAD_FOLDER = "src/data/input"


@app.route('/score', methods=['POST'])
def score() -> Response:

    try:

        data = request.files

        values=['target','e3_ligase','protac','e3_ligase_ligand','target_ligand']
        for value in values:
            if value not in data:
                raise ValueError (f"Missing {value} in form data")
        
        if 'label' not in request.form:
            raise ValueError ("Missing label in form data")
        
        label = request.form.get('label')
        if label not in ['0', '1']:
            raise ValueError ("Invalid label value. Must be 0 or 1")
        
        if 'reqId' not in request.form:
            raise ValueError ("Missing reqId")
        reqId=request.form.get('reqId')
        


        json_data = {}
        file_paths = []
        for value in values:
            file = request.files[value]
            if file: 
                json_data[value] = reqId+'_'+file.filename
                file_path = os.path.join(UPLOAD_FOLDER, reqId+'_'+file.filename)
                file_paths.append(file_path)
                file.save(file_path)
            else:
                raise ValueError (f"Missing {value} in form data")
            
        json_data['label'] = int(label)

        if not json_data:
            return Response("No data provided", status=400)
        
        with open(f"src/data/input/{reqId}_input.json", "w") as f:
            json.dump(json_data, f)

        if(os.path.exists(f"src/data/input/{reqId}_input.json")):
            file_paths.append(f"src/data/input/{reqId}_input.json")
        
        # command = f'cd src && python3 case_study.py'
        # result = subprocess.run(command, shell=True, capture_output=True, text=True)

        result=main()

        return jsonify({
            "message": "Inference completed successfully",
            "output": result
        }), 200
    
    except ValueError as ve:
        traceback.print_exc()
        return jsonify({"message":"Inference failed","error": f"{ve}"}), 400
    
    # except Exception as e:
    #     traceback.print_exc()
    #     return jsonify({"message":"Inference failed","error": f"{e}"}), 500
    
    finally:
        if len(os.listdir(UPLOAD_FOLDER)) > 0:
            for file in os.listdir(UPLOAD_FOLDER):
                if file == "features":
                    continue
                file_path = os.path.join(UPLOAD_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    
@app.route('/health/<sample>', methods=['POST'])
def samplescore(sample) -> Response:
 
    if sample == "hi":
        date=datetime.now().strftime("%H:%M:%S")
        return f"Hello {date}"
    else:
        return jsonify({'error':"Unauthorized access"})



if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=5000,debug=False,use_reloader=False)
    app.run(host='0.0.0.0',port=5000,debug=True)