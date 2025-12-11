# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns

import mesa

# Import Cell Agent and OrthogonalMooreGrid
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

import matplotlib.pyplot as plt

from mesa.batchrunner import batch_run

# Add function for model level collection
def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B


class MoneyAgent(CellAgent):
    """An agent with fixed initial wealth."""

    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.wealth = 1
        self.steps_not_given = 0  # NEW: how many steps since last time this agent received money?

    def move(self):
        self.cell = self.cell.neighborhood.select_random_cell()

    def give_money(self):
        cellmates = [a for a in self.cell.agents if a is not self]

        if self.wealth > 0 and cellmates:
            other_agent = self.random.choice(cellmates)
            other_agent.wealth += 1
            self.wealth -= 1
            # Reset counter for the receiver
            other_agent.steps_not_given = 0
        else:
            # If we didn't give money to anyone this step, increment our own counter
            self.steps_not_given += 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = OrthogonalMooreGrid(
            (width, height), torus=True, capacity=20, random=self.random
        )
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={
                "Wealth": "wealth",
                "Steps_not_given": "steps_not_given",  # NEW
            },
        )
        self.running = True

        # Create agents
        agents = MoneyAgent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents),
        )

    def step(self):
        # Collect data each step
        self.datacollector.collect(self)
        self.agents.shuffle_do("move")
        self.agents.do("give_money")

model = MoneyModel(200, 10, 10)
for _ in range(200):
    model.step()

# Extract MoneyModel data in a Pandas dataframe
gini = model.datacollector.get_model_vars_dataframe()

# Extract MoneyAgent data in a Pandas dataframe
agent_wealth = model.datacollector.get_agent_vars_dataframe()

# Transform the data to a long format
agent_wealth_long = agent_wealth.T.unstack().reset_index()
agent_wealth_long.columns = ["Step", "AgentID", "Variable", "Value"]
print(agent_wealth_long.head(3))

# Plot the average wealth over time
g = sns.lineplot(data=agent_wealth_long, x="Step", y="Value", errorbar=("ci", 95))
g.set(title="Average wealth over time")

plt.show()

# save the model data (stored in the pandas gini object) to CSV
gini.to_csv("model_data.csv")

# save the agent data (stored in the pandas agent_wealth object) to CSV
agent_wealth.to_csv("agent_data.csv")

# ----------------------
# BATCH RUN: parameter sweep
# ----------------------

# Sweep over different population sizes
params = {
    "n": [50, 100, 200, 400],  # number of agents
    "width": 10,
    "height": 10,
}

if __name__ == "__main__":
    # On Windows, keeping number_processes=1 avoids multiprocessing issues.
    results = batch_run(
        MoneyModel,
        parameters=params,
        iterations=5,          # 5 runs per parameter combination
        max_steps=100,         # each run lasts 100 steps
        number_processes=1,    # keep simple and safe on Windows
        data_collection_period=1,  # collect data every step
        display_progress=True,
    )

    # Turn the list of dicts into a dataframe
    results_df = pd.DataFrame(results)
    print(f"Batch results shape: {results_df.shape}")
    print(results_df.head())

    # Keep only one row per (RunId, Step) by filtering a single agent
    gini_df = results_df[results_df["AgentID"] == 1]

    # Lineplot: Gini over time, colored by population size n
    plt.figure()
    g = sns.lineplot(
        data=gini_df,
        x="Step",
        y="Gini",
        hue="n",
        errorbar=("ci", 95),
    )
    g.set(
        title="Gini coefficient over time (batch runs)",
        ylabel="Gini",
        xlabel="Step",
    )
    plt.tight_layout()
    plt.show()
