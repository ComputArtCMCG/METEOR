from designer import PPO
import operator,functools
import mpi4py.MPI as MPI
import time,yaml
from typing import *

class Runner():
    def __init__(self,yaml_file,comm:MPI.COMM_WORLD):
        with open(yaml_file,'r') as f:
            kwargs = yaml.load(f,Loader=yaml.FullLoader)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        self.designer = PPO(self.rank,self.comm_size,**kwargs) 

    def acting(self):
        ### actor_run
        self.comm.Barrier()
        self.designer.acting()
        self.comm.Barrier()

    def activity_estimate(self):
        self.comm.Barrier()
        if self.designer.activity_stat is not None:
            self.designer.activity_stat.estimate_all()
        else:
            pass
        self.comm.Barrier()

    def cal_adv(self):
        all_smis,all_raw_rewards,n_qed,n_sascore,n_affinity,n_trajs,square_sum,num_sum,n_states = self.designer.raw_trajectory_processor.process_raw_trajectories()
        combined_smis = self.comm.gather((all_smis,self.rank),root=0)
        combined_raw_rewards = self.comm.gather((all_raw_rewards,self.rank),root=0)
        total_n_qed = self.comm.gather(n_qed,root=0)
        total_n_sascore = self.comm.gather(n_sascore,root=0)
        total_n_affinity = self.comm.gather(n_affinity,root=0)
        total_trajs = self.comm.gather(n_trajs,root=0)
        total_square_sum = self.comm.gather(square_sum,root=0)
        total_num_sum = self.comm.gather(num_sum,root=0)
        total_n_states = self.comm.gather(n_states,root=0)
        if self.rank ==0 :
            combined_smis.sort(key=lambda x:x[1])
            combined_raw_rewards.sort(key = lambda x:x[1])
            combined_smis = functools.reduce(operator.concat,[i[0] for i in combined_smis])
            combined_raw_rewards = functools.reduce(operator.concat,[i[0] for i in combined_raw_rewards])
            total_n_qed = sum(total_n_qed)
            total_n_sascore = sum(total_n_sascore)
            total_n_affinity = sum(total_n_affinity)
            total_trajs = sum(total_trajs)
            total_square_sum = sum(total_square_sum)
            total_num_sum = sum(total_num_sum)
            total_n_states = sum(total_n_states)
            adv_mean,adv_std = self.designer.raw_trajectory_processor.cal_advantage_and_update_memory(combined_smis,combined_raw_rewards,\
                                                                                                    total_n_qed,total_n_sascore,\
                                                                                                    total_n_affinity,total_trajs,total_square_sum,total_num_sum,total_n_states,\
                                                                                                        self.designer.round)
            return adv_mean,adv_std
        else:
            return None,None


    def ppo_loop(self):
        self.acting()
        self.activity_estimate()
        adv_mean,adv_std = self.cal_adv()
        if self.rank == 0:
            dataloader = self.designer.prepare_loader(adv_mean,adv_std)
            self.designer.training(dataloader)
            for idx in range(1,self.comm_size):
                self.comm.send('OK',dest=idx,tag=self.designer.round)
        else:
            while not self.comm.Iprobe(source=0,tag=self.designer.round):
                time.sleep(1.0)
        self.comm.Barrier()
        self.designer.synchronize()
        self.comm.Barrier()

    def main(self):
        while time.time() - self.designer.start_time <= self.designer.time_limit:
            self.ppo_loop()
            if self.rank == 0:
                print('--------------------------------------------------------------------------------------')
    
if __name__ == '__main__':
    from rdkit import RDLogger
    import argparse
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',required=True,default='generate',choices=['generate','yaml'],\
        help='running for generation through reinforcement learning ,yaml for generating default settings')
    parser.add_argument('-y','--yaml',required=True)
    args = parser.parse_args()
    if args.mode == 'yaml':
        PPO.write_yaml(args.yaml)
    elif args.mode == 'generate':
        comm = MPI.COMM_WORLD
        runner = Runner(args.yaml,comm)
        runner.main()