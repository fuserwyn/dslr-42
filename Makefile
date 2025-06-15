PYTHON = python

# === High-level Targets ===
all: describe histogram scatter pair

# === Data Analysis ===
describe:
	$(PYTHON) describe.py dataset_train.csv

bonus-describe:
	$(PYTHON) describe.py dataset_train.csv --bonus

histogram:
	$(PYTHON) histogram.py dataset_train.csv

histogram-save:
	$(PYTHON) histogram.py dataset_train.csv --save

scatter:
	$(PYTHON) scatter_plot.py dataset_train.csv

pair:
	$(PYTHON) pair_plot.py dataset_train.csv

pair-manual:
	$(PYTHON) pair_plot.py dataset_train.csv --manual

# === Training Modes ===
train:
	$(PYTHON) logreg_train.py dataset_train.csv

train-sgd:
	$(PYTHON) logreg_train.py dataset_train.csv --mode sgd

train-minibatch:
	$(PYTHON) logreg_train.py dataset_train.csv --mode minibatch

# === Prediction ===
predict:
	$(PYTHON) logreg_predict.py dataset_test.csv weights.csv

# === Setup and Cleanup ===
setup:
	bash setup_project.sh

clean:
	rm -f *.png *.csv __pycache__/*.pyc
	rm -rf __pycache__ .venv best_scatter_pairs histograms

.PHONY: all describe bonus-describe histogram histogram-save scatter pair pair-manual train train-sgd train-minibatch predict setup clean
