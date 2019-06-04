
install: 
	python setup.py install --user
uninstall:
	python setup.py remove --user


test: test_vrp

test_vrp_cargo:
	python -m unittest src.examples.vrp.test.VRPTestSuit

test_vrp:
	python -m unittest src.examples.vrp.test.VRPTestSuit
run_vrp:
	python -m src.examples.vrp.VRP

clean: 
	rm -rf __pycache__
	rm -rf GeneticAlgorithm.egg-info
	rm -rf src/__pycache__
	rm -rf src/examples/__pycache__
	rm -rf src/examples/.system
	rm -rf src/utils/__pycache__


