# UsingNeuralNetworksToPredictBetaInCryptoAndCrossAssetMarkets
Before running code, make sure you have the following:

- CRSP Dataset downloaded into data folder in directory
- Fama-French 5 Factor data downloaded into data folder in directory
- Bitcoin price data from Coinbase via FRED downloaded into data folder in directory
- Ethereum price data from Coinbase via FRED downloaded into data folder in directory
- Litecoin price data from Coinbase via FRED downloaded into data folder in directory
- Bitcoin Cash price data from Coinbase via FRED downloaded into data folder in directory
- Nominal broad US Dollar Index price data from FRED downloaded into data folder in directory (proxy for fiat) 
- Vanguard Energy ETF price data downloaded into data folder in directory (proxy for energy)
- Credit Suisse NASDAQ Gold FLOWS103 Price Index data from FRED downloaded into data folder in directory (proxy for gold)

To run the code, follow the following steps in a bash terminal:

1. Create virtual environment and activate by running:
```bash
python3 -m venv venv && source venv/bin/activate
```
2. Upgrade pip and install all required libraries into virtual environment:
```bash
pip install --upgrade pip && pip install -r requirements.txt
```
3. Run the code:
```bash
python main.py
```

Outputs should appear in outputs folder and/or in your terminal