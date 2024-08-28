from states import Stack,mol_to_nx,tensorize_single_nxgraph,finished_to_prob_mask_single
from networks import ActorCritic
from env import largest_ring,convert_radical_electrons_to_hydrogens
from rdkit import Chem
from typing import *
import copy,torch
from tqdm import tqdm

NOT_ALLOWED_PATTENS = [
    Chem.MolFromSmarts("*=[#6]=*"),
    Chem.MolFromSmarts("[#8]-[#8]"),
    Chem.MolFromSmarts("[#16]-[#16]"),
    Chem.MolFromSmarts("[#9,#17,#35,#53]=,#*"),
    Chem.MolFromSmarts("[r]#[r]"), #环内三键
    Chem.MolFromSmarts("[r3,r4]=[r3,r4]"), #三 or 四元环内的双键    
    Chem.MolFromSmarts("[#16&R&D3,4]"),
    # following are aromatic bridge ring
    Chem.MolFromSmarts('a1(*2)a@2aaaa1'),
    Chem.MolFromSmarts('a1@2aa@1aaa@2'),
    Chem.MolFromSmarts('a1(*2)aaaa@2a1'),   #C1(C2)=CC=CC2=C1
    Chem.MolFromSmarts('a1(**2)aaaa@2a1'),
    Chem.MolFromSmarts('a1(***2)aaaa@2a1'),
    Chem.MolFromSmarts('*1(*2)=**=*@2*=*1'),
    Chem.MolFromSmarts('a1(**2)aaa@2aa1'), 
    Chem.MolFromSmarts('a1(***2)aaa@2aa1'), 
    #five-membered ring
    Chem.MolFromSmarts('a1(*2)a@2aaa1'), 
    Chem.MolFromSmarts('a1@2aa@1aa2'), 
    Chem.MolFromSmarts('a1(*2)aa@2aa1'), 
    Chem.MolFromSmarts('a1(**2)aa@2aa1'), 
    Chem.MolFromSmarts('a1(***2)aa@2aa1'), 
]
HALOGEN = ['Cl','F','Br','I']

class ExpertEnv():
    def __init__(self,max_action = 50):
        self.possible_atoms = ['C','N','O','S','F','I', 'Cl','Br']
        self.possible_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE]
        self.max_action = 2 * max_action
        self.max_atom = max_action + len(self.possible_atoms)
        self.focus_stack = Stack()
        self.focus_stack.push(0)
        self.finished_nodes = []
        self.finish = False
        self.start_mol = Chem.RWMol(Chem.MolFromSmiles('C'))
        self.mol = copy.deepcopy(self.start_mol)
        self.mol_old = None
        self.counter = 0
    
    def step(self,action:torch.LongTensor):
        """
        Perform a given action
        :param action: LongTensor with shape [1,3]
        :param action_type:
        """
        ### init
        action = action.squeeze().tolist()
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        total_atoms = self.mol.GetNumAtoms()

        ### take action  action : [atom_index,bind_type,stop]
        ### in action, atom_index is start from 0

        current_focus:int = self.focus_stack.items[-1]
        if action[2] == 0 :
            if action[0] >= total_atoms:  #when =, there is a new node 
                self._add_atom(action[0] - total_atoms)  # add new node
                action[0] = total_atoms  # new node id
                self._add_bond(current_focus,action)  # add new edge
            else:
                self._add_bond(current_focus,action)  # add new edge
        else:
            if current_focus not in self.get_terminal_olefin():
                self.finished_nodes.append(current_focus)
                self.focus_stack.pop()
            else:
                pass

        self.process_intermediate(action)
        self.counter += 1
    
    def process_intermediate(self,action): 
        # check chemical validity first (implicitly modify the formal charge of candidate N atoms to +1)
        # then check allowed actions (this method need self.mol to be sanitized)
        if  self.check_chemical_validity(action) and self.check_not_allowed_action():
            if action[2] == 0:
                if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
                    if self.mol.GetAtomWithIdx(self.mol_old.GetNumAtoms()).GetSymbol() not in HALOGEN:
                        self.focus_stack.push(self.mol_old.GetNumAtoms())
                    else:  #halogen atoms are masked when produced
                        self.finished_nodes.append(self.mol_old.GetNumAtoms())
                elif  self.mol.GetNumAtoms() == self.mol_old.GetNumAtoms() and self.mol.GetNumBonds()-self.mol_old.GetNumBonds()>0:
                    pass
                else:
                    self.mol = self.mol_old
            else: #action[2] == 1
                pass
        else:
            self.mol = self.mol_old
        Chem.SanitizeMol(self.mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

        ### calculate terminal rewards
        terminate_condition = self.mol.GetNumAtoms() >= self.max_atom - len(self.possible_atoms) or self.counter >= self.max_action or \
            self.focus_stack.is_empty()
        
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

    def _add_bond(self, current_focus,action):
        '''
        :param action: [first_node, second_node, bong_type_id]
        :return:
        '''
        bond_type = self.possible_bond_types[action[1]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(current_focus, action[0])
        
        if not bond and current_focus!=action[0]:
            self.mol.AddBond(current_focus, action[0], order=bond_type)

    def check_chemical_validity(self,action):
        """
        Checks the chemical validity of the mol object. 
        Specially for Nitro and Quaternary ammonium and Onium salt
        :return: True if chemically valid, False otherwise
        """
        if action[2] == 1:
            return True
        else:
            current_focus = self.focus_stack.items[-1] 
            if action[0] >= self.mol.GetNumAtoms():
                end_idx = self.mol.GetNumAtoms()-1
            else:
                end_idx = action[0]
            
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

    def get_terminal_olefin(self) -> List[int]:
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True,canonical=False)
        m = Chem.MolFromSmiles(s)
        matches = m.GetSubstructMatches(Chem.MolFromSmarts(r'[#6H2;$([#6]=[#6])]'))
        return [i[0] for i in matches]  #index of terminal olefin C

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
        self.focus_stack.__init__()
        self.focus_stack.push(0)
        self.finished_nodes = []
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
            (unfinished_mols,focus,finished_nodes) = self.gather_unfinished_mols_focus()
            (mol_features,focus,prob_mask) = self.inputs2tensor(unfinished_mols,focus,finished_nodes)
            (action,probs,qvalue) = self.actor_critic.action_qvalue_from_mols(mol_features,focus,prob_mask)
            unfinished_envs = [env for env in self.envs if not env.finish]
            self.envs_step(unfinished_envs,action)

    def gather_unfinished_mols_focus(self) -> Tuple[List,List[int],List[List[int]]]:
        unfinished_mols = [env.mol for env in self.envs if not env.finish]
        focus = [env.focus_stack.items[-1] for env in self.envs if not env.finish]
        finished_nodes = [env.finished_nodes for env in self.envs if not env.finish]
        return (unfinished_mols,focus,finished_nodes)

    def inputs2tensor(self,mols:List,focus:List[int],finished_nodes:List[List[int]]):
        #return tensors stacked with shape [batch, bla, bla]
        outs =  list(map(self._input2tensor,zip(mols,focus,finished_nodes)))
        # stack the tensors
        mol_features = [i[0] for i in outs]
        mol_features = (torch.stack([i[0] for i in mol_features],dim=0),
                        torch.stack([i[1] for i in mol_features],dim=0),
                        torch.stack([i[2] for i in mol_features],dim=0),
                        torch.stack([i[3] for i in mol_features],dim=0),
                        )                   
        focus = torch.LongTensor([i[1] for i in outs]).view([-1,1])
        prob_mask = torch.stack([i[2] for i in outs],dim=0)
        assert mol_features[0].shape[0] == focus.shape[0] == prob_mask.shape[0]
        return (mol_features,focus,prob_mask)

    def _input2tensor(self,args):
        mol,focus,finished_nodes = args
        mol_graph = mol_to_nx(mol)
        mol_features = tensorize_single_nxgraph(mol_graph)
        prob_mask = finished_to_prob_mask_single(mol_graph,focus,finished_nodes)
        return (mol_features,focus,prob_mask)

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
