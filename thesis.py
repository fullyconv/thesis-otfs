import torch
import matplotlib.pyplot as plt


class Agent:
    """Generic moving entity with Torch vectors."""
    def __init__(self, pos, vel, name=""):
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.vel = torch.tensor(vel, dtype=torch.float32)
        self.name = name

    def step(self):
        """Advance position by one time step."""
        self.pos += self.vel


class Environment:
    def __init__(self, M, N, Snr, RxList, TxList, ObjList, device='cpu'):
        self.M = M
        self.N = N
        self.snr = Snr
        self.device = device

        # store all agents as fields (lists of Agent objects)
        self.RxList = [self._to_device(rx) for rx in RxList]
        self.TxList = [self._to_device(tx) for tx in TxList]
        self.ObjList = [self._to_device(obj) for obj in ObjList]

    def _to_device(self, agent):
        """Ensure tensors are on the same device."""
        agent.pos = agent.pos.to(self.device)
        agent.vel = agent.vel.to(self.device)
        return agent

    def step(self):
        """Advance all agents by one time step."""
        for lst in [self.RxList, self.TxList, self.ObjList]:
            for agent in lst:
                agent.step()

    def get_report(self):
        """Print all agent positions."""
        print("=== Environment Report ===")
        for i, rx in enumerate(self.RxList):
            print(f"Rx {i}: pos={rx.pos.tolist()} vel={rx.vel.tolist()}")
        for i, tx in enumerate(self.TxList):
            print(f"Tx {i}: pos={tx.pos.tolist()} vel={tx.vel.tolist()}")
        for i, obj in enumerate(self.ObjList):
            print(f"Obj {i}: pos={obj.pos.tolist()} vel={obj.vel.tolist()}")

    def plot(self):
        """Plot current snapshot."""
        plt.figure(figsize=(6, 6))
        # Plot each category
        if self.TxList:
            tx_pos = torch.stack([tx.pos for tx in self.TxList])
            plt.scatter(tx_pos[:, 0].cpu(), tx_pos[:, 1].cpu(), c='red', label='Tx')
        if self.RxList:
            rx_pos = torch.stack([rx.pos for rx in self.RxList])
            plt.scatter(rx_pos[:, 0].cpu(), rx_pos[:, 1].cpu(), c='blue', label='Rx')
        if self.ObjList:
            obj_pos = torch.stack([obj.pos for obj in self.ObjList])
            plt.scatter(obj_pos[:, 0].cpu(), obj_pos[:, 1].cpu(), c='green', label='Obj')

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Environment Snapshot")
        plt.grid(True)
        plt.show()



if __name__ == '__ main __':
    # Define agents
    RxList = [Agent(pos=[1, 2], vel=[1, 2], name="Rx1")]
    TxList = [Agent(pos=[1, 0], vel=[1, 0], name="Tx1")]
    ObjList = [Agent(pos=[2, 3], vel=[0.5, -1], name="Obj1")]

    # Create environment
    env = Environment(M=64, N=64, Snr=10,
                    RxList=RxList, TxList=TxList, ObjList=ObjList)


    for i in range(10):
        print("Before step:")
        env.get_report()
        # Plot environment
        env.plot()

        # Step once
        env.step()


