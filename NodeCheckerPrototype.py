import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd


class SimpleCheckSeq(Chare):
    # Not strictlly necessary
    def __init__(self):
        self.myWorkList = []
    
    @coro
    def initList(self, myWorkList):
        self.status = Future()
        self.myWorkList = myWorkList
        self.status.send(1)

    @coro
    def collectXferStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.sum)

    @coro
    def check(self, reduceCallback):
        # Set 'val' to the LOGICAL OR of the truth value for each node integer in self.myWorkList
        val = True if 0 in self.myWorkList else False
        # Leave this line alone
        self.reduce(reduceCallback, val , Reducer.logical_or)
