# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns

import mesa

import matplotlib.pyplot as plt 

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B

class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model, ethnicity):
        super().__init__(model)
        self.wealth = 1
        self.ethnicity = ethnicity

    def give_money(self, receivers):
        if self.wealth > 0 and len(receivers) > 0:
            other_agent = self.random.choice(receivers)
            other_agent.wealth += 1
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n):
        super().__init__()
        self.num_agents = n

        # Create a list of our different ethnicities
        ethnicities = ["Green", "Blue", "Mixed"]

        # Create agents
        MoneyAgent.create_agents(
            model=self,
            n=self.num_agents,
            ethnicity=self.random.choices(ethnicities, k=self.num_agents),
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Gini": compute_gini,
                # average wealth per ethnicity using AgentSet.agg
                "MeanWealth_Green": lambda m: m.agents.select(
                    lambda a: a.ethnicity == "Green"
                ).agg("wealth", np.mean),
                "MeanWealth_Blue": lambda m: m.agents.select(
                    lambda a: a.ethnicity == "Blue"
                ).agg("wealth", np.mean),
                "MeanWealth_Mixed": lambda m: m.agents.select(
                    lambda a: a.ethnicity == "Mixed"
                ).agg("wealth", np.mean),
            },
            agent_reporters={"Wealth": "wealth", "Ethnicity": "ethnicity"},
        )

    def step(self):
        self.datacollector.collect(self)
        # AgentSet of all mixed agents (used as receivers)
        mixed_agents = self.agents.select(lambda a: a.ethnicity == "Mixed")
        # Create dictionary of agents groupby
        grouped_agents = model.agents.groupby("ethnicity")
        for ethnic, similars in grouped_agents:
            if ethnic in ["Blue", "Green"]:
                # Blue & Green give only to Mixed
                similars.shuffle_do("give_money", mixed_agents)
            else:
                # Mixed can give to everyone
                similars.shuffle_do("give_money", self.agents)

# Run the model
model = MoneyModel(100)
for _ in range(20):
    model.step()

# --------- Agent-level plot (as before) ----------
agent_data = model.datacollector.get_agent_vars_dataframe()
palette = {"Green": "green", "Blue": "blue", "Mixed": "purple"}
g = sns.histplot(
    data=agent_data,
    x="Wealth",
    hue="Ethnicity",
    discrete=True,
    palette=palette,
)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents")
plt.show()

# --------- Example of using the agg-based model data ----------
model_data = model.datacollector.get_model_vars_dataframe()
print(model_data.tail())  # shows Gini and mean wealth per ethnicity over time

