import asyncio
import copy
import random
import time
import traceback
from collections import defaultdict

import aiohttp
import bittensor as bt
# from CliqueAI.chain.snapshot import Snapshot
from CliqueAI.graph.client import get_graph
# from CliqueAI.graph.codec import GraphCodec
# from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
# from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator
# from CliqueAI.selection.miner_selector import MinerSelector
from CliqueAI.selection.problem_selector import ProblemSelector
# from CliqueAI.transport.axon_requester import AxonRequester
# from common.base import validator_int_version, validator_version
# from common.base.validator import BaseValidatorNeuron
# from common.base.wandb_logging.model import WandbRunLogData


async def get_validator_graph():

    wallet = bt.wallet('cdy', 'hk1')
    
    problem_selector = ProblemSelector(
        miner_selector=None,
    )
    
    problem = problem_selector.select_problem()
    print(problem)
    time_limit = problem_selector.select_time_limit()
    print(time_limit)
    
    try:
        graph = await get_graph(
            wallet=wallet,
            netuid=83,
            label=problem.label,
            time_limit=time_limit,
            number_of_nodes_min=problem.vertex_range.min,
            number_of_nodes_max=problem.vertex_range.max,
            number_of_edges_min=problem.edge_range.min,
            number_of_edges_max=problem.edge_range.max,
        )
    
    except Exception as e:
        bt.logging.error(f"Error fetching graph: {e}")
        await asyncio.sleep(30)
        return
    
    bt.logging.info(f"Selected problem: {problem}")
    
    print(graph.adjacency_list)
    
if __name__ == "__main__":
    asyncio.run(get_validator_graph())
