PYTHON = python

all: describe histogram scatter pair

describe:
	$(PYTHON) describe.py dataset_train.csv

histogram:
	$(PYTHON) histogram.py dataset_train.csv

scatter:
	$(PYTHON) scatter_plot.py dataset_train.csv

pair:
	$(PYTHON) pair_plot.py dataset_train.csv

train:
	$(PYTHON) logreg_train.py dataset_train.csv

predict:
	$(PYTHON) logreg_predict.py dataset_test.csv weights.csv

clean:
	rm -f *.png *.csv __pycache__/*.pyc

.PHONY: all describe histogram scatter pair train predict clean
