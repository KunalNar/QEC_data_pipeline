.DEFAULT_GOAL := help

PYTHON ?= python

.PHONY: help check inventory syndromes qasmbench google gold-syndromes gold-qasmbench ml-syndromes run-syndrome-qasm run train test

help:
	@echo "Student pipeline commands"
	@echo ""
	@echo "  make check      Verify object-store and PostgreSQL connections"
	@echo "  make inventory  List the supplied course data files"
	@echo "  make syndromes   Build the syndrome Silver and evidence tables"
	@echo "  make qasmbench   Build the QASMBench Silver and evidence tables"
	@echo "  make google      Build the Google QEC Silver and evidence tables"
	@echo "  make gold-syndromes  Load the chosen syndrome Gold tables into PostgreSQL"
	@echo "  make gold-qasmbench  Load QASMBench circuit Gold tables into PostgreSQL"
	@echo "  make ml-syndromes    Export the syndrome ML table from PostgreSQL Gold"
	@echo "  make run-syndrome-qasm  Rebuild both sources through Gold and syndrome ML"
	@echo "  make run        Run your pipeline after implementing the stages"
	@echo "  make train      Train/evaluate models from the required ML input tables"
	@echo "  make test       Run automated tests"

check:
	$(PYTHON) -m quantum_lake_student.cli check

inventory:
	$(PYTHON) -m quantum_lake_student.cli inventory

syndromes:
	$(PYTHON) -m quantum_lake_student.cli prepare-syndromes

qasmbench:
	$(PYTHON) -m quantum_lake_student.cli prepare-qasmbench

google:
	$(PYTHON) -m quantum_lake_student.cli prepare-google

gold-syndromes:
	$(PYTHON) -m quantum_lake_student.cli load-syndromes-gold

gold-qasmbench:
	$(PYTHON) -m quantum_lake_student.cli load-qasmbench-gold

ml-syndromes:
	$(PYTHON) -m quantum_lake_student.cli build-ml-syndromes

run-syndrome-qasm:
	$(PYTHON) -m quantum_lake_student.cli run-syndrome-qasm

run:
	$(PYTHON) -m quantum_lake_student.cli run

train:
	$(PYTHON) -m quantum_lake_student.cli train

test:
	$(PYTHON) -m pytest
