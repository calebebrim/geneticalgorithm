
install: 
	python setup.py install --user
uninstall:
	python setup.py remove --user






clean: 
	rm -rf __pycache__
	rm -rf GeneticAlgorithm.egg-info
	rm -rf src/__pycache__
	rm -rf src/examples/__pycache__
	rm -rf src/examples/.system
	rm -rf src/utils/__pycache__


skills: 
	python -m src.examples.skills_optimization

maximization: 
	python -m src.examples.sum_maximization

minimization:
	python -m src.examples.sum_minimization
