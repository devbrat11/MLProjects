
from flask import Flask,request, jsonify
from DataAnalyzer import MlModel

app = Flask(__name__)            

mlAnalyzer = MlModel()

@app.route("/")                
def status(): 
    return "Ml App is running using Python and Flask..."  

@app.route('/data', methods=['GET'])                  
def getData():
    data = mlAnalyzer.getData()
    return data

@app.route('/model', methods=['GET'])                  
def createMlModel():
    modelDetails = mlAnalyzer.createModel()                    
    return jsonify(modelDetails)   

                                   
if __name__ == "__main__":        # on running python app.py
    app.run(debug=True)                     # run the flask app