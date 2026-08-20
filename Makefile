PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python
SCRIPT = main.py

all:
	$(PYTHON_VENV) $(SCRIPT)

venv:
	$(PYTHON) -m venv $(VENV)
	$(SOURCE) venv/bin/activate
	$(PIP) install pandas scikit-learn

run:
	$(PYTHON_VENV) $(SCRIPT)

clean:
	rm -rf $(VENV)
	rm -rf results