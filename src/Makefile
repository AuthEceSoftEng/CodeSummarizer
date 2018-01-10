VENV=${HOME}/.code_summarizer_venv
PYTHON=python3
IPYTHON=ipython3
PIP=pip3

virtual-env:
	$(PYTHON) -m venv $(VENV)

base-deps:
	. $(VENV)/bin/activate && \
	$(PIP) install -r requirements.txt

setup: virtual-env base-deps

update: virtual-env
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade -r requirements.txt

clean:
	rm -rf $(VENV)
