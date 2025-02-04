#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = byol_project
PYTHON_VERSION = y
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 byol
	isort --check --diff --profile black byol
	black --check --config pyproject.toml byol

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml byol


## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



.PHONY: byol_pre_training
byol_pre_training:
	$(PYTHON_INTERPRETER) -m byol.modeling.train pre_train


.PHONY: byol_fine_tuning
byol_fine_tuning:
	$(PYTHON_INTERPRETER) -m byol.modeling.train fine_tune


.PHONY: baseline_training
baseline_training:
	$(PYTHON_INTERPRETER) -m byol.modeling.train baseline


.PHONY: byol_prediction
byol_prediction:
	$(PYTHON_INTERPRETER) -m byol.modeling.predict byol


.PHONY: baseline_prediction
baseline_prediction:
	$(PYTHON_INTERPRETER) -m byol.modeling.predict baseline


.PHONY: run
run:
	$(PYTHON_INTERPRETER) -m byol.modeling.train pre_train
	$(PYTHON_INTERPRETER) -m byol.modeling.train fine_tune
	$(PYTHON_INTERPRETER) -m byol.modeling.train baseline
	$(PYTHON_INTERPRETER) -m byol.modeling.predict byol
	$(PYTHON_INTERPRETER) -m byol.modeling.predict baseline




#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
