# MARLENA: Explain Multilabel Classifiers

This project aims to address the multi-label black-box outcome explanation problem. Introducing MARLENA (**M**ulti-l**a**bel **R**u**l**e-based **E**xpla**NA**tions)!

```
@inproceedings{panigutti2019explaining,
  title={Explaining multi-label black-box classifiers for health applications},
  author={Panigutti, Cecilia and Guidotti, Riccardo and Monreale, Anna and Pedreschi, Dino},
  booktitle={International Workshop on Health Intelligence},
  pages={97--110},
  year={2019},
  organization={Springer}
}
```
You can find MARLENA python library [here](https://github.com/CeciPani/MARLENA).

## Running MARLENA
0.**Install the requirements.txt listed packages**

1. **Install *multilabelexplanations***: In order to run MARLENA you first have to locally install the pyhton module *multilabelexplanations*. You can do this by running the following command into the module directory:
 ~~~~
 $ cd multilabelexplanations
 $ pip install .
 ~~~~
2. **Run the python scripts**: you have to run the code python folde in the listed order:  
  *1_prepare_datasets.py*  
  *2_train_blackbox.py* 	  
  *3_pairwise_distances.py*    
  *4_global_decision_tree.py*
  
3. **Run MARLENA experiments py script** the script to run the experiments is in the *python* folder
  ~~~~
 $ python experiments_tuned_optimized.py [dataset_name] [black_box_name]
 ~~~~
4. **Analyze the results** using the *Analysis* jupyter notebooks 

## Outputs 

You can find experiments results into the *output* folder.
