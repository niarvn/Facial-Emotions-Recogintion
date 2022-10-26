# Facial-Emotions-Recogintion
This project's purpose is to use CNN to detect facial of 7 basic human emotion. We implemented on PyCharm. Using Flask framework to create to built a website provide
user interface to get input (frontface camera). With the help of OpenCV we can easily detect the face and then using that frame we apply CNN model for that frame to detect. 
## Steps
* Make a virtual environment using:
```
pip install pipenv
```

* Uninstall previous flask if facing errors
```
pip3 uninstall flask
```
* Install libraries:
```
python -m pip install flask
python -m pip install opencv-python
python -m pip install tensorflow
python -m pip install keras
```
* Install requirement: 
```
pip install -r requirements.txt
```
* Run project: 
```
set FLASK_APP=app.py
$env:FLASK_APP="app.py"
```
```
python3 -m flask run
```
Deploy address: http://127.0.0.1:5000/
