VENV=${HOME}/.code_summarizer_venv
PYTHON=python3
IPYTHON=ipython3
PIP=pip3

virtual-env:
	$(PYTHON) -m venv $(VENV)

base-deps:
	. $(VENV)/bin/activate && \
	$(PIP) install -r requirements.txt

setup: virtual-env base-deps nltk-setup

nltk-setup: virtual-env base-deps
	. $(VENV)/bin/activate && \
	ipython3 -m preprocessor.nltk_setup

update: virtual-env
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade -r requirements.txt

clean:
	rm -rf $(VENV)
