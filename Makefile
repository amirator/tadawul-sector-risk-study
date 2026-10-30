.PHONY: all refresh test notebook
all:        ## run full pipeline on cached data
	python run.py
refresh:    ## re-download data from Yahoo Finance, then run
	python run.py --refresh
test:
	python -m pytest -q
notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/tadawul_risk_walkthrough.ipynb
