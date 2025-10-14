from ax import Client, RangeParameterConfig

# 1. Initialize the Client.
client = Client()

# 2. Configure where Ax will search.
client.configure_experiment(
    name="booth_function",
    parameters=[
        RangeParameterConfig(
            name="x1",
            bounds=(-10.0, 10.0),
            parameter_type="float",
        ),
        RangeParameterConfig(
            name="x2",
            bounds=(-10.0, 10.0),
            parameter_type="float",
        ),
    ],
)

# 3. Configure a metric for Ax to target (see other Tutorials for adding constraints,
# multiple objectives, tracking metrics etc.)
client.configure_optimization(objective="-1 * booth")

# 4. Conduct the experiment with 20 trials: get each trial from Ax, evaluate the
# objective function, and log data back to Ax.
for _ in range(20):
    # Use higher value of `max_trials` to run trials in parallel.
    for trial_index, parameters in client.get_next_trials(max_trials=1).items():
        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2
                + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2
            },
        )

# 5. Obtain the best-performing configuration; the true minimum for the booth
# function is at (1, 3).
best = client.get_best_parameterization()
print(best)
