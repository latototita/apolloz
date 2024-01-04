from flask import Flask
import script
app = Flask(__name__)

@app.route('/')
def home():
    return "I am alive"

if __name__=="__main__":
  os.system("python3 script.py &")
  app.run(host="0.0.0.0", port=80)
