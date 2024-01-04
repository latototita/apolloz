from flask import Flask
app = Flask(__name__)
from script import run_trading_bot
symbol='XAGUSDm'
timeframe='1h'


@app.route('/')
def home():
    return "I am alive"
@app.route('/index')
def index():
    print('Start')
    run_trading_bot()
    print('Stop')
    return "I am alive"

if __name__=="__main__":
  os.system("python script.py &")
  app.run(host="0.0.0.0", port=80)
