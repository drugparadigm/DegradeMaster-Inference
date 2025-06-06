# DegradeMaster
Inference code of ISMB/ECCB'25 submitted paper "Accurate PROTAC targeted degradation prediction with DegradeMaster"

* Install Docker engine.
* Add the files/folders required only for inference in src directory

* Use request.form to access inputs from the request
* Make sure request.form has reqId
* Save or Dump inputs in src/data/input with filename as {reqId}-input.json
* If the request body are files then dump those files in src/data/input with reqId as prefix along with filebane
* If you need to access the input files(i.e. in the code file) then retrieve them from src/data/input
* To access other inputs in main function, get reqId using:
```
reqId=request.form.get('reqId')
```
then access via {reqId}-input.json


* Add the model checkpoints in src/checkpoints dir
* Make sure your checkpoints are saved as torch.save(model.state_dict())
* If the code downloads any pre-trained models, please let us know.


* Generate a yaml file with your conda environment from dgx with the command:
```
conda env export > environment.yaml
```
* Make sure the yaml file contains packages required only for inference

* Add the packages that are hard to install with the installation commands in additional-softwares.sh

* Rename the filenames and environment names in Dockerfile.

* Rename the function(main) in api.py with your main inference function.


* Raise all the possible Exceptions and catch them in api.py
* Raise the Exceptions so the users should be able to understand the error and rectify it 

* Make sure to rename the files/folders in api.py whereever required.
* Do not have any nested json as output
* Run api.py with the above changes:
```
gunicorn api:app -b 0.0.0.0:5000 --workers=1 --threads=5  --access-logfile -
```

* If there are no errors then create a docker image with the command:
```
    docker build -t <image-name> .
```
* Run the image with:
```
    docker run -p 5000:5000 <image-name> (If your PC has no GPU)

    docker run --gpus all -p 5000:5000 <image-name> (If your PC has GPU)
```

#### Test the /hi endpoint
* Use Python’s requests library to send an HTTP POST (with an empty JSON body) to your service and print the result:
```
import requests
from requests.structures import CaseInsensitiveDict

url = "http://localhost:5000/health/hi"

headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"
headers["Content-Type"] = "application/json"
headers["Content-Length"] = "0"


resp = requests.post(url, headers=headers)
print(resp)
# Print status, headers and pretty-printed JSON body
print("Status code:", resp.status_code)
print("Headers:", resp.headers)
print("Body:",resp.text)
#print(json.dumps(resp.json(), indent=2)) #uncomment if it is a json response
```
#### Test the /score endpoint
```
import requests

url = "http://0.0.0.0:5000/score"  
data = {
    <Please provide the required input format for the API request body.Include reqId.>
}

# Send as multipart/form-data
response = requests.post(url, data=data)

print(response.text)

```
 
