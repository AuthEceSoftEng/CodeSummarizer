VENV=${HOME}/.code_summarizer_venv
PYTHON=python3
IPYTHON=ipython3
PIP=pip3

virtual-env:
	$(PYTHON) -m venv $(VENV)

base-deps:
	. $(VENV)/bin/activate && \
	$(PIP) install -r requirements.txt

setup: virtual-env base-deps fix-pytagcloud

fix-pytagcloud: virtual-env base-deps
	2to3 -w ${HOME}/.code_summarizer_venv/lib/python3.5/site-packages/pytagcloud

update: virtual-env
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade -r requirements.txt

clean:
	rm -rf $(VENV)
