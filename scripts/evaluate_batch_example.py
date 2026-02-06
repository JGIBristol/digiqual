import pandas as pd
from digiqual.core import SimulationStudy
from digiqual.sampling import generate_lhs

# --- CONFIGURATION ---

# 1. Update the Command for the 'scripts' folder
#    We add "addpath('scripts');" so MATLAB can find your .m file
SOLVER_CMD = "matlab -batch \"addpath('scripts'); run_sim('{input}', '{output}')\""

# 2. File Paths
INPUT_CSV = "sim_inputs.csv"
OUTPUT_CSV = "sim_results.csv"

# 3. Define Ranges
INPUT_RANGES = {
    "Length": (0.0, 10.0),
    "Angle": (-45.0, 45.0)
}
OUTCOME_VAR = "Signal"

print("\n=== STARTING DIGIQUAL RELIABILITY WORKFLOW ===\n")

# --- STEP 1: INITIALISATION ---
study = SimulationStudy(
    input_cols=list(INPUT_RANGES.keys()),
    outcome_col=OUTCOME_VAR
)

# --- STEP 2: INITIAL DESIGN ---
print("--- ITERATION 1: Generating Initial Design (LHS) ---")

# 1. Convert Dictionary to the DataFrame format required by generate_lhs
# sampling.py requires columns: "Name", "Min", "Max"
input_df = pd.DataFrame([
    {"Name": k, "Min": v[0], "Max": v[1]}
    for k, v in INPUT_RANGES.items()
])

# 2. Call the function with the correct arguments
# Note: The function signature is generate_lhs(n, vars_df, seed)
n_start = 20
initial_samples = generate_lhs(
    n=n_start,
    vars_df=input_df
)

# --- STEP 3: RUN SIMULATION (BATCH 1) ---
# Run the external loop
study.evaluate_batch(
    samples=initial_samples,
    command_template=SOLVER_CMD,
    input_path=INPUT_CSV,
    output_path=OUTPUT_CSV
)

# --- STEP 4: ADAPTIVE LOOP ---
max_iterations = 3

for i in range(max_iterations):
    print(f"\n--- DIAGNOSTICS CHECK (Iteration {i+1}) ---")
    diag = study.diagnose()
    print(diag)

    if diag.empty:
        print(">> Error: No diagnostic data available (Simulation likely failed).")
        break

    # Then continue with your normal check
    if diag['Pass'].all():
        # ...
        print("\n>>> SUCCESS: Model Validation Passed! <<<")
        break

    print("\n>> WARNING: Issues detected. Refining design...")

    # Generate ~10 new points based on gaps/uncertainty
    new_samples = study.refine(n_points=10)

    if new_samples.empty:
        print("No further refinement suggested. Stopping.")
        break

    print(f"--- ITERATION {i+2}: Running {len(new_samples)} New Simulations ---")

    # Run the loop ONLY for the new points
    study.evaluate_batch(
        samples=new_samples,
        command_template=SOLVER_CMD,
        input_path=INPUT_CSV,
        output_path=OUTPUT_CSV
    )

# --- STEP 5: FINAL ANALYSIS ---
print("\n=== FINAL ANALYSIS ===")

try:
    # Run Probability of Detection (PoD)
    study.pod(
        poi_col="Length",
        threshold=8.0,
        n_boot=1000
    )

    # Visualise and Save
    print("Generating plots...")
    study.visualise(show=True, save_path="final_report")
    print("\nWorkflow Complete. Plots saved.")

except ValueError as e:
    print(f"\nAnalysis Failed: {e}")
