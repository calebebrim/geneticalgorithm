
install: 
	python setup.py install --user
uninstall:
	python setup.py remove --user
test:
	python -m unittest src/tests/VRPTestSuit.py

run:
	python -m src.example.VRP