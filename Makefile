
install: 
	python setup.py install --user
uninstall:
	python setup.py remove --user


test: test_vrp

test_vrp:
	python -m unittest src.examples.vrp.test.VRPTestSuit
run_vrp:
	python -m src.examples.vrp.VRP

