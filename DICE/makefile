# Chose with set from mvtec to generate counterfactuals for

DATASET_NAME ?= bottle
PYTHON := ./dice-env/bin/python

run:
	@echo "🔧 Using dataset: $(DATASET_NAME)"
	@echo "DATASET_NAME=$(DATASET_NAME)" > .env
	@echo "🧠 Training classifier..."
	@$(PYTHON) train.py
	@echo "🎯 Generating DiCE counterfactuals..."
	@$(PYTHON) dice_cf.py

# Generate counterfactuals for every set from mvtec
# DATASET_NAME ?= bottle
# PYTHON := ./dice-env/bin/python

# run:
# 	@echo "🔧 Using dataset: $(DATASET_NAME)"
# 	@echo "DATASET_NAME=$(DATASET_NAME)" > .env
# 	@echo "🧠 Training classifier..."
# 	@$(PYTHON) train.py
# 	@echo "🎯 Generating DiCE counterfactuals..."
# 	@$(PYTHON) dice_cf.py

# run_all:
# 	@for dataset in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do \
# 		make run DATASET_NAME=$$dataset; \
# 	done
