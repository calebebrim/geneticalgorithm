# Genetic Algotithm Experiments


Documentation for Genetic Algorithm : 
- GA
- GAU

GAU is the defaut functions used on "GA.__init__". Only modify if you are confident what are you doing.



## Usage:

To run the examples you must use python module notation: 
python -m path.to.filename

### Example 1 - Maximization:
    Sum 2 numbers, genetich will optimize those numbers to get highest sum.

    python -m example.sum_maximization
### Example 2 - Minimization: 
    Sum 2 numbers, genetich will optimize those numbers to get lowest sum.
    
    python -m example.sum_minimization
### Example 2 - Using Threads:
    Sum 2 numbers, genetich will optimize those numbers to get highest sum. 
    This will use threads, IDK what it is so slow so help me if you have some sugestion about how to implement it wisely. 

    I think that it should be used only when your fitness function takes too much to run.

    python -m example.sum_paralel.py

    