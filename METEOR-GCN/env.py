import copy,torch,pickle
from typing import *
import rdkit.Chem as Chem 
from rdkit import RDLogger
from networks import ActorCritic
from states import TrajectoryState,Trajectory,Stack,mol_to_nx,tensorize_single_nxgraph
from states import mol_to_nx
from states import HALOGEN,NOT_ALLOWED_PATTENS,largest_ring

RDLogger.DisableLog('rdApp.*')
class MoleculeEnv():
    def __init__(self,start_mol,max_action = 50):

        self.possible_atoms = ['C','N','O','S','F','I', 'Cl','Br']
        self.possible_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE]
        self.max_action = max_action
        self.max_atom = max_action + len(self.possible_atoms)   #len(possible_atoms)
        ###start_mol (eg.scaffold to keep) 
        if start_mol is not None:
            self.start_mol = Chem.RWMol(Chem.MolFromSmiles(start_mol))
            Chem.SanitizeMol(self.start_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            self.mol = copy.deepcopy(self.start_mol)
        else:
            self.start_mol = Chem.RWMol(Chem.MolFromSmiles('C'))
            #self._add_atom(0)  #always start from one 'C'
            self.mol = copy.deepcopy(self.start_mol)
        
        self.reward_step = 0.01
        self.counter = 0
        self.finish = False

        ###recording the trajectory
        self.trajectory = []
    
    def step(self,action:torch.LongTensor,probs,qvalue):
        """
        Perform a given action
        :param action: LongTensor with shape [1,4]
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,-1 if otherwise
        """
        ### init
        action_input = copy.deepcopy(action)
        action = action.squeeze().tolist()
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        total_atoms = self.mol.GetNumAtoms()

        ### take action  action : [atom_index,atom_index,bind_type,stop]
        ### in action, atom_index is start from 0

        if action[3]==0: # not stop
            if action[1] >= total_atoms:  #when =, there is a new node 
                self._add_atom(action[1] - total_atoms)  # add new node
                action[1] = total_atoms  # new node id
                self._add_bond(action)  # add new edge
            else:
                self._add_bond(action)  # add new edge
        else: # stop
            pass

        # get reward
        step_reward = self.step_reward_calculate(action)
        # get state
        state = self.get_state(action_input,step_reward)
        state.set_probs(probs)
        state.set_qvalue(qvalue)
        self.trajectory.append(state)

    def step_reward_calculate(self,action):
        ### end is decided from action
        ### calculate intermediate rewards
        self.counter += 1
        reward_step = 0.00
        if self.check_chemical_validity(action) and self.check_not_allowed_action():
            if action[3] == 0:
                if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
                    new_atom = self.mol.GetAtomWithIdx(self.mol_old.GetNumAtoms())
                    if new_atom.GetSymbol() in ['Cl','Br','I']:
                        reward_step = -0.01
                elif  self.mol.GetNumAtoms() == self.mol_old.GetNumAtoms() and self.mol.GetNumBonds()-self.mol_old.GetNumBonds()>0:
                    reward_step = 0.01 #successfully produce a ring
                else:
                    reward_step = -0.2 # edge exist
                    self.mol = self.mol_old  #reject update
            else:
                reward_step = 0.0  # finish
        else:
            reward_step = -0.2  # invalid action
            self.mol = self.mol_old  #reject update

        Chem.SanitizeMol(self.mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        
        ### calculate terminal rewards
        terminate_condition = self.mol.GetNumAtoms() >= self.max_atom - len(self.possible_atoms) or \
                               self.counter >= self.max_action or action[3] == 1
        if terminate_condition :
            self.finish = True # end of episode
        ### use stepwise reward
        return reward_step

    def reset(self):
        self.mol = copy.deepcopy(self.start_mol)
        self.mol_old = None
        self.finish = False
        self.counter = 0
        self.trajectory = []
        
    def _add_atom(self, atom_type_id):
        """
        Adds an atom
        :param atom_type_id: atom_type id
        :return:
        """
        # assert action.shape == (len(self.possible_atom_types),)
        # atom_type_idx = np.argmax(action)
        atom_type = self.possible_atoms[atom_type_id]
        atom_to_add = Chem.Atom(atom_type)
        #atom_to_add.SetFormalCharge(atom_type[1])
        self.mol.AddAtom(atom_to_add)

    def _add_bond(self, action):
        '''
        :param action: [first_node, second_node, bong_type_id]
        :return:
        '''
        # GetBondBetweenAtoms fails for np.int64
        bond_type = self.possible_bond_types[action[2]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[0]), int(action[1]))
        
        if not bond and action[0]!=action[1]:
            self.mol.AddBond(int(action[0]), int(action[1]), order=bond_type)

    def check_chemical_validity(self,action):
        """
        Checks the chemical validity of the mol object. Existing mol object is
        not modified. Radicals pass this test.
        :return: True if chemically valid, False otherwise
        """
        if action[3] == 1:
            return True
        else:
            current_focus = action[0] 
            if action[1] >= self.mol.GetNumAtoms():
                end_idx = self.mol.GetNumAtoms()-1
            else:
                end_idx = action[1]
            
            focus_atom = self.mol.GetAtomWithIdx(current_focus)
            end_atom = self.mol.GetAtomWithIdx(end_idx)
            if focus_atom.GetSymbol() == 'N': 
                bonds = [(atom.GetIdx(),current_focus) for atom in self.mol.GetAtomWithIdx(current_focus).GetNeighbors()]
                valence = [int(self.mol.GetBondBetweenAtoms(*bond).GetBondType()) for bond in bonds]
                if sum(valence) == 4 and 3 not in valence and end_atom.GetSymbol() in ['C','O','N'] and action[1] == 0 :
                    self.mol.GetAtomWithIdx(current_focus).SetFormalCharge(1)
                    if end_atom.GetSymbol() == 'O':
                        self.mol.GetAtomWithIdx(end_idx).SetFormalCharge(-1)
                    if end_atom.GetSymbol() == 'N':
                        end_bonds = [(atom.GetIdx(),end_idx) for atom in self.mol.GetAtomWithIdx(end_idx).GetNeighbors()]
                        end_valence = [int(self.mol.GetBondBetweenAtoms(*bond).GetBondType()) for bond in end_bonds]
                        if sum(end_valence) == 4 and 3 not in end_valence:
                            self.mol.GetAtomWithIdx(end_idx).SetFormalCharge(1)

            s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
            m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
            if m:
                return True
            else:
                return False

    def check_not_allowed_action(self):
        """ 
        check after the current action, wether the current mol is OK,including
        :not a over 7-member ring or not reasonable structures (*=C=* or O-O or S-S triple bond in ring)  
        :return: True if is OK, False otherwise
        """
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True,canonical=False)
        m = Chem.MolFromSmiles(s)
        ring_info = largest_ring(m)
        forbid_structure_matches = [len(m.GetSubstructMatches(pattern)) > 0 for pattern in NOT_ALLOWED_PATTENS]
        if ring_info > 7 or any(forbid_structure_matches):
            return False
        else:
            return True
    
    def check_valency(self) -> bool: 
        """
        Checks that no atoms in the mol have exceeded their possible
        valency
        :return: True if no valency issues, False otherwise
        """
        try:
            Chem.SanitizeMol(self.mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except ValueError:
            return False

    # TODO(Bowen): check if need to sanitize again
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)

    # TODO(Bowen): check if need to sanitize again
    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return m

    def get_state(self,action_input,step_reward):
        state = TrajectoryState(mol_to_nx(self.mol_old),action_input,step_reward)
        return state
    
    def get_trajectory(self):
        if not self.check_valency():
            smi = 'XXX'
        else:
            mol = self.get_final_mol()
            smi = Chem.MolToSmiles(mol)
        return Trajectory(self.trajectory,smi)

class MolEnvWarpper():
    def __init__(self,envs:List[MoleculeEnv],actor_critic:ActorCritic,raw_traj_file:str):
        self.envs = envs
        self.actor_critic = actor_critic
        self.raw_traj_file = raw_traj_file
    
    def acting(self):
        self._acting()
        self.reset_and_to_pickle()
        # all env have finished their generation process
        # write down the generation trajectory
    
    def _acting(self):
        while not all([env.finish for env in self.envs]): 
            unfinished_mols = self.gather_unfinished_mols()
            mol_features = self.inputs2tensor(unfinished_mols)
            (action,probs,qvalue) = self.actor_critic.action_qvalue_from_mols(mol_features)
            unfinished_envs = [env for env in self.envs if not env.finish]
            self.envs_step(unfinished_envs,action,probs,qvalue)
    
    def gather_unfinished_mols(self) -> List:
        return [env.mol for env in self.envs if not env.finish]
    
    def inputs2tensor(self,mols:List):
        #return tensors stacked with shape [batch, bla, bla]
        mol_features =  list(map(self._input2tensor,mols))
        # stack the tensors
        mol_features = (torch.stack([i[0] for i in mol_features],dim=0),
                        torch.stack([i[1] for i in mol_features],dim=0),
                        torch.stack([i[2] for i in mol_features],dim=0),
                        torch.stack([i[3] for i in mol_features],dim=0),
                        )                   
        return mol_features
    
    def _input2tensor(self,mol):
        mol_graph = mol_to_nx(mol)
        mol_features = tensorize_single_nxgraph(mol_graph)
        return mol_features
    
    def envs_step(self,unfinished_envs,action,probs,qvalue):
        """
        :action with shape [mols,3]
        :probs with shape [mols,3]
        :qvalue with shape [mols,1]
        """        
        for (env,a,p,q) in zip(unfinished_envs,action,probs,qvalue):
            env.step(a.unsqueeze(0),p.unsqueeze(0),q.unsqueeze(0))    
    
    def reset_and_to_pickle(self):
        with open(self.raw_traj_file,'ab') as f:
            for env in self.envs:
                traj = env.get_trajectory()
                pickle.dump(traj,f)
                env.reset()

def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m
