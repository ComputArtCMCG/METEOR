from networks import ActorCritic
from states import mol_to_nx,tensorize_single_nxgraph,largest_ring
from env import convert_radical_electrons_to_hydrogens
from rdkit import Chem
from typing import *
import copy,torch
from tqdm import tqdm

class ExpertEnv():
    def __init__(self,max_action = 50):
        self.possible_atoms = ['C','N','O','S','F','I', 'Cl','Br']
        self.possible_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE]
        self.max_action = 2 * max_action
        self.max_atom = max_action + len(self.possible_atoms)
        self.finished_nodes = []
        self.finish = False
        self.start_mol = Chem.RWMol(Chem.MolFromSmiles('C'))
        self.mol = copy.deepcopy(self.start_mol)
        self.mol_old = None
        self.counter = 0
    
    def step(self,action:torch.LongTensor):
        """
        Perform a given action
        :param action: LongTensor with shape [1,4]
        :param action_type:
        """
        ### init
        action = action.squeeze().tolist()
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        total_atoms = self.mol.GetNumAtoms()

        ### take action  action : [atom_index,bind_type,stop]
        ### in action, atom_index is start from 0

        if action[3]==0:
            if action[1] >= total_atoms:  #when =, there is a new node 
                self._add_atom(action[1] - total_atoms)  # add new node
                action[1] = total_atoms  # new node id
                self._add_bond(action)  # add new edge
            else:
                self._add_bond(action)  # add new edge
        else:
            self.finish = True
        self.counter += 1
        self.process_intermediate()
    
    def process_intermediate(self): 
        # check chemical validity first (implicitly modify the formal charge of candidate N atoms to +1)
        # then check allowed actions (this method need self.mol to be sanitized)
        if  self.check_valency():
            if self.mol.GetNumAtoms()+self.mol.GetNumBonds()-self.mol_old.GetNumAtoms()-self.mol_old.GetNumBonds()>0:
                pass
            else:            
                self.mol = self.mol_old
        else: 
            self.mol = self.mol_old
        
        Chem.SanitizeMol(self.mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        terminate_condition = self.mol.GetNumAtoms() >= self.max_atom - len(self.possible_atoms) or self.counter >= self.max_action 
        if terminate_condition:
            self.finish = True

    def get_final_smi(self) -> Union[str,None]:
        ###valid check
        if not self.check_valency():
            smi = None
        else:
            # final mol object where any radical electrons are changed to bonds to hydrogen
            final_mol = self.get_final_mol()
            smi = Chem.MolToSmiles(final_mol, isomericSmiles=True)
        return smi
    
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

    def _add_bond(self,action):
        '''
        :param action: [first_node, second_node, bong_type_id,end]
        :return:
        '''
        bond_type = self.possible_bond_types[action[2]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[0]), int(action[1]))
        
        if not bond and action[0]!=action[1]:
            self.mol.AddBond(int(action[0]), int(action[1]), order=bond_type)

    def check_chemical_validity(self):
        """
        Checks the chemical validity of the mol object. Existing mol object is
        not modified. Radicals pass this test.
        :return: True if chemically valid, False otherwise
        """
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
        if m:
            return True
        else:
            return False

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

    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)

    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return m
    
    def reset(self):
        self.mol = copy.deepcopy(self.start_mol)
        self.mol_old = None
        self.finish = False
        self.counter = 0
        
class MolEnvWarpper():
    def __init__(self,envs:List[ExpertEnv],actor_critic:ActorCritic):
        self.envs = envs
        self.actor_critic = actor_critic

    def acting(self):
        self._acting()
        all_smis = self.gather_smis()
        return all_smis

    def _acting(self):
        while not all([env.finish for env in self.envs]): 
            unfinished_mols = self.gather_unfinished_mols_focus()
            mol_features = self.inputs2tensor(unfinished_mols)
            (action,probs,qvalue) = self.actor_critic.action_qvalue_from_mols(mol_features)
            unfinished_envs = [env for env in self.envs if not env.finish]
            self.envs_step(unfinished_envs,action)

    def gather_unfinished_mols_focus(self) -> Tuple[List,List[int],List[List[int]]]:
        unfinished_mols = [env.mol for env in self.envs if not env.finish]
        return unfinished_mols

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

    def envs_step(self,unfinished_envs:List[ExpertEnv],action):
        """
        :action with shape [mols,3]
        :probs with shape [mols,3]
        :qvalue with shape [mols,1]
        """        
        for (env,a) in zip(unfinished_envs,action):
            env.step(a.unsqueeze(0))

    def gather_smis(self):
        all_smis = [env.get_final_smi() for env in self.envs]
        for env in self.envs:
            env.reset()
        return all_smis

    def all_finish(self):
        return all([env.finish for env in self.envs])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num',required=True,type=int)
    parser.add_argument('-p','--parm',required=True,type=str)
    parser.add_argument('-b','--batch_size',type=int,default=50)
    parser.add_argument('-o','--out',required=True,type=str)
    args = parser.parse_args()
    total_num,parm_file = args.num,args.parm
    envs_num = args.batch_size
    actor_critic = ActorCritic(64,3,device=torch.device('cuda'))
    actor_critic.set_device(torch.device('cuda'))
    actor_critic.load_state_dict(torch.load(parm_file,map_location=torch.device('cuda')))
    envs_warpper = MolEnvWarpper([ExpertEnv() for _ in range(envs_num)],actor_critic)
    assert total_num % envs_num == 0
    loop_count = int(total_num/envs_num)
    with open(args.out,'w',encoding='UTF-8') as f:
        for _ in tqdm(range(loop_count)):
            all_smis = envs_warpper.acting()
            for s in all_smis:
                if s is not None:
                    f.write(s+'\n')
                else:
                    f.write('XXX\n')

