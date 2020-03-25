import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(r'C:\Users\BitMining\3D Objects\clova\CRAFT-pytorch-master\CRAFT-pytorch-master\image\do.jpg','rb')})