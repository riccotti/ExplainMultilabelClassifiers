# MARLENA: Explain Multilabel Classifiers

This project aims to address the multi-label black-box outcome explanation problem. Introducing MARLENA (**M**ulti-l**a**bel **R**u**l**e-based **E**xpla**NA**tions)!

## Running MARLENA

1. **Install *multilabelexplanations***: In order to run MARLENA you first have to locally install the pyhton module *multilabelexplanations*. You can do this by running the following command into the module directory:
 ~~~~
 $ cd multilabelexplanations
 $ pip install .
 ~~~~
2. **Run the jupyter notebooks**: you have to run the code in the jupyter notebooks in the listed order:  
  *1_prepare_datasets.ipynb*  
  *2_train_blackbox.ipynb* 	  
  *3_pairwise_distances.ipynb*    
  *4_global_decision_tree.ipynb*
  
3. **Run MARLENA experiments py script** the script to run the experiments is in the *python* folder
  ~~~~
 $ python experiments_tuned_optimized.py [dataset_name] [black_box_name]
 ~~~~
4. **Analyze the results** using the *Analysis* jupyter notebooks 

## Outputs 

You can find experiments results into the *output* folder.
