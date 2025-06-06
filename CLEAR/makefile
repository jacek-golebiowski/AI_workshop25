DATASET_NAME ?= bottle
PYTHON := ./clear-env/bin/python

run:
	@echo "🔧 Using dataset: $(DATASET_NAME)"
	@echo "DATASET_NAME=$(DATASET_NAME)" > .env
	@echo "📦 Training AE..."
	@$(PYTHON) train_ae.py
	@echo "🧠 Training classifier..."
	@$(PYTHON) train.py
	@echo "🔍 Running CLEAR counterfactuals..."
	@$(PYTHON) clear_image_cf.py
