install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort .
	black .
	
lint:
	pylint *.py

all: install format lint
