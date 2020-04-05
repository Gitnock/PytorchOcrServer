# IMG FROM LOCAL
import requests

# url = "http://localhost:5000/predict"

files={"file": open(r'image URL','rb')}

resp = requests.post(url,files=files)

print( resp.json())


