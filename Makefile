PYTHON = python3
VENV = venv
PY = $(VENV)/bin/python

setup:            ## cria o ambiente virtual e instala dependências
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

train:            ## treina os modelos e gera métricas + submissões do Kaggle
	$(PY) train.py

demo:             ## abre a demo Gradio em http://127.0.0.1:7860
	$(PY) app.py

web:              ## abre a versão Flask em http://127.0.0.1:5000
	$(PY) flask_app.py

publish:          ## publica modelo e demo no Hugging Face (rode `hf auth login` antes)
	$(PY) publish_hf.py

clean:
	rm -rf $(VENV) __pycache__
