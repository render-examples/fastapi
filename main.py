from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_live_gold_price():
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        price = data['chart']['result'][0]['meta']['regularMarketPrice']
        return float(price)
    except Exception as e:
        return 2415.50

@app.get("/")
def get_gold_signal():
    live_price = get_live_gold_price()
    tp = round(live_price + 10.0, 2)
    sl = round(live_price - 5.0, 2)
    
    return {
        "status": "success",
        "symbol": "XAUUSD",
        "price": live_price,
        "signal": "BUY",
        "entry": live_price,
        "tp": tp,
        "sl": sl,
        "timeframe": "M1"
    }
