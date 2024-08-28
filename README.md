
# **METEOR**

### METEOR: _**M**olecular-**E**xploration **T**hrough **E**xtra-**O**bjective **R**einforcement_

De novo drug design offers an appealing approach for discover novel leads at the early phase of drug discovery. Its key advantage lies in its ability to explore a much broader dimension of chemical space, without being confined to the knowledge of existing compounds. So far numerous generative models have been described in the literature, which have completely redefined the concept of de novo drug design. <br>

We have developed a graph-based generative model within a reinforcement learning framework, namely METEOR (Molecular Exploration Through Extra-Objective Reinforcement). The backend agent of METEOR is based on the well-established GCPN model. To ensure the overall quality of the generated molecular graphs, we implemented a set of rules to identify and exclude undesired substructures. Importantly, METEOR is designed to conduct multi-objective optimization, i.e. simultaneously optimizing binding affinity, drug-likeness, and synthetic accessibility of the generated molecules under the guidance of a special reward function. 

### Usage
1. Setup dependencies
```bash
conda env create -f meteor.yaml
conda activate meteor
```
2. Using METEOR <br>
We have created a demo folder (GBA_design) in METEOR-DFS and METEOR-GCN, which includes a protein file (prepared_receptor.pdb), a crystal ligand file (crystal_ligand.sdf) and a run config file for METEOR (design.yaml). The config file contains many parameters, you can use it as a template for your own task. Important parameters are listed as below:
   - activity_lower_bound,activity_upper_bound: define how to calculate activity reward. Since METEOR uses PLANET as the backend for predicting binding affinity, one can use PLANET to estimate some active molecules to know the range of PLANET's output on this target protein.
   - qed_lower_bound and qed_upper bound: define how to calculate drug-likeness reward.
   - sascore_lower_bound and sascore_upper_bound: define how to calculate synthetic accessbility reward.
   - activity_cutoff, qed_cutoff, sascore_cutoff: has no effect on result, only let you know the current statement of reinforcement learning.
   - ligand_sdf and protein_pdb: exactly PLANET paramenters. (center_x,centor_y,center_z can be null if ligand_sdf is provided)
   - design_log: the path of output file (including SMILES genenrated during the METEOR run)
   - time_limit: define the time of METEOR run (unit: day) <br>

```bash
cd METEOR-GCN
mpiexec -np 10 python3.6 mpirun.py -m generate -y GBA_design/design.yaml
```
The command will rise 10 parallel jobs running METEOR. We test the code on two 2080Ti GPU (11G memory), 5 jobs on each GPU will not cause out of CUDA memory error. Besides, each job will cost about 4GB system memory. <br>
Wait until finish and you can analysis generated molecules listed in GBA_design/design.log.


