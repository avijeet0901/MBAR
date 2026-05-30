# MBAR

This code applies the Multistate Bennett Acceptance Ratio (MBAR) in both the temperature and multidimensional collective variable (CV) spaces. The biased potential used in the simulation is harmonic, as in umbrella sampling, and requires two inputs: the location of the harmonic potential and the spring constant.

# Package requirement
```
pip3 install numpy
pip3 install os
pip3 install argparse
pip3 install pymbar
```
# Usage
The help options can be printed out with the -h option. and the code can be executed by providing all the necessary input:

python mbar_weights.py -f input_file.dat -target_temp 272.0 -stride 10 -N_CV 2 -periodic YES NO -periodicity "[-pi, pi]

The input_file file should have the following format, including all the COLVAR files:
/location/to/file/COLVAR CV1 CV2 ... K1 K2 ... temperature
...
Two input demo files for 2- and 4-CV are included in the repository. 

where the file COLVAR should have the following columns in sequence: time step, CV1, CV2, ..., the potential energy of the system. The potential energy should not contain the biased energy contribution. 

The target_temp argument specifies the temperature at which the weights are computed. NOTE: This temperature should not be too far from the sampled temperature range for better weight estimates. 

The stride argument specifies the interval at which to read data points from the COLVAR files. 

The N_CV argument specifies the number of biased CVs in the simulations. 

-periodic argument asks if a CV is periodic, and should be provided as a string of YES and NO inputs. The default is NO for all the CVs. 

-periodicity is required if any of the arguments of periodic is YES. Here, the periodicity needs to be defined in square brackets for all YES in sequence. 

-------
The error in the free-energy estimate, or in estimates of other properties, can be evaluated in two ways: (i) by recomputing the weights for each bootstrap sample, or (ii) by performing bootstrap analysis on the sampled data while reusing the weights computed from the complete dataset, thereby reducing the computational cost. 

--------
Feel free to contact me at avijeetkulshrestha@gmail.com if you find any bugs or need help running the code

# Citations
Please cite the following paper if you are using the code or any segment of code:

```
@article{kulshrestha2022finite,
  title={Finite temperature string method with umbrella sampling using path collective variables: application to secondary structure change in a protein},
  author={Kulshrestha, Avijeet and Punnathanam, Sudeep N and Ayappa, K Ganapathy},
  journal={Soft Matter},
  volume={18},
  number={39},
  pages={7593--7603},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```
# Future updates 
A faster approach to computing errors in the unbiased properties will be implemented in the future. 

Best Regards.  
Avijeet Kulshrestha



