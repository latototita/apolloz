from flask import Flask
import os

app = Flask(__app__)

@app.route('/')
def home():
  return "I am alive"


if __name__=="__main__":
  os.system("python script.py &")
  app.run(host="0.0.0.0", port=80)
