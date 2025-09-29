import pybamm
import yaml

def load_parameters(config_path):
    """
    Load battery parameters from a YAML/JSON config file and return a PyBaMM ParameterValues object.
    Also computes any derived parameters needed for the model.
    """
    # Load the YAML or JSON file
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    # Create a ParameterValues object, starting from a base chemistry if needed
    # We'll start from PyBaMM's default parameter set for the given chemistry, if available.
    chem = params.get("chemistry", None)
    if chem is not None and chem.lower().startswith("graphite_lmo"):
        # Use a generic Li-ion parameter set as baseline (e.g., Chen2020 for NMC, or a custom LMO set if available)
        param_values = pybamm.ParameterValues("Marquis2019")  # Marquis2019 as a baseline Li-ion set
    else:
        param_values = pybamm.ParameterValues("Marquis2019")  # default to a known set
    
    # Override parameters with values from config
    # Negative electrode parameters
    neg = params["NegativeElectrode"]
    param_values.update({
        "Maximum concentration in negative electrode [mol.m-3]": neg["max_concentration"],
        "Negative particle radius [m]": neg["particle_radius"],
        "Negative electrode diffusivity [m2.s-1]": neg["diffusivity"],
        "Negative electrode exchange-current density [A.m-2]": neg["exchange_current_density"],
        "Initial concentration in negative electrode [mol.m-3]":
            neg["initial_stoichiometry"] * neg["max_concentration"],
        # SEI parameters
        "SEI growth rate constant [m.s-1]": neg.get("sei_growth_rate", 0),
        "Initial SEI thickness [m]": neg.get("sei_initial_thickness", 0),
        "SEI resistivity [Ohm.m]": neg.get("sei_resistivity", 0),
        # Cracking parameters (if any)
        "Crack growth rate constant [m-1]": neg.get("crack_growth_rate", 0),
        "SEI charge per crack area [C.m-2]": neg.get("sei_charge_per_area", 0),
    }, check_already_exists=False)
    
    # Positive electrode parameters
    pos = params["PositiveElectrode"]
    param_values.update({
        "Maximum concentration in positive electrode [mol.m-3]": pos["max_concentration"],
        "Positive particle radius [m]": pos["particle_radius"],
        "Positive electrode diffusivity [m2.s-1]": pos["diffusivity"],
        "Positive electrode exchange-current density [A.m-2]": pos["exchange_current_density"],
        "Initial concentration in positive electrode [mol.m-3]":
            pos["initial_stoichiometry"] * pos["max_concentration"],
        "Initial volume fraction in positive electrode": pos.get("initial_active_fraction", 1.0),
    }, check_already_exists=False)
    
    # Mn dissolution / LAM parameter (we will implement via current-driven LAM in PyBaMM)
    if pos.get("Mn_dissolution_rate", None) is not None:
        k_diss = pos["Mn_dissolution_rate"]
        # Define a function for positive electrode LAM vs current (use a constant loss rate when current flows)
        def lam_positive(i, T):
            # We'll cause active material loss during cycling (i != 0). No loss during rest (i=0).
            if abs(i) < 1e-6:
                return 0.0
            else:
                return -k_diss  # fractional change per second (negative sign for decay)
        param_values.update({
            "Positive electrode current-driven LAM rate": lam_positive
        }, check_already_exists=False)
    else:
        k_diss = 0.0
    
    # Electrolyte and other resistances
    if "Electrolyte" in params:
        elyte = params["Electrolyte"]
        param_values.update({
            "Initial concentration in electrolyte [mol.m-3]": elyte.get("c0", 1000),
            "Electrolyte diffusivity [m2.s-1]": elyte.get("diffusivity", 5e-10),
            "Electrolyte conductivity [S.m-1]": elyte.get("conductivity", 1.0),
        }, check_already_exists=False)
    if "Resistance" in params:
        res = params["Resistance"]
        param_values.update({
            "Contact resistance [Ohm]": res.get("R_contact", 0.0)
        }, check_already_exists=False)
    
    # Return the ParameterValues object
    return param_values

def build_spm_model(param_values):
    """
    Build a PyBaMM Single Particle Model (SPM) with degradation submodels enabled.
    """
    # Define model options to include degradation mechanisms
    options = {
        "SEI": "solvent-diffusion limited",          # SEI growth model (diffusion-limited)
        "SEI on cracks": "true",                     # enable SEI on crack surfaces
        "SEI porosity change": "true",               # account for porosity changes due to SEI
        "loss of active material": "current-driven", # use current-driven LAM (for cathode Mn dissolution)
        "particle mechanics": ("swelling and cracking", "none"),  
        # Particle mechanics: "swelling and cracking" for anode (neg), none or swelling-only for cathode.
        # (We assume cathode structural change is handled by LAM, not mechanical cracking in this model.)
    }
    model = pybamm.lithium_ion.SPM(options=options)
    # Set any necessary initial conditions for new states:
    # PyBaMM will automatically add SEI thickness variable and set initial from "Initial SEI thickness".
    # For crack-derived variables, ensure initial crack surface area is zero:
    # The PyBaMM "SEI on cracks" submodel typically uses the crack progression from particle mechanics.
    # We use current-driven LAM for positive electrode active material loss, already set via param.
    return model

def simulate_battery(param_values, N_cycles=1, cycle_current=1.0, plot=False):
    """
    Simulate the battery for a given number of cycles and return results.
    - N_cycles: number of charge/discharge cycles to simulate.
    - cycle_current: magnitude of the current for cycling (C-rate relative to 1C).
    - plot: if True, generate plots of the results.
    """
    # Build the model
    model = build_spm_model(param_values)
    
    # Create an experiment: define the sequence of steps for each cycle.
    # We'll do a constant-current discharge and charge for each cycle, possibly with rests.
    C_rate = cycle_current  # use as 1C equivalent current
    cycle_protocol = [
        f"Discharge at {C_rate}C until 2.5 V",   # discharge to cutoff voltage
        "Rest for 10 minutes",
        f"Charge at {C_rate}C until 4.2 V",
        "Hold at 4.2 V until C/50",             # top off charge until current tapers
        "Rest for 10 minutes"
    ]
    # For multiple cycles, repeat the protocol N_cycles times
    experiment = pybamm.Experiment(cycle_protocol * N_cycles)
    
    # Initialize and run the simulation
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param_values)
    print("Running simulation for", N_cycles, "cycle(s) ...")
    sol = sim.solve()
    
    # Extract results of interest
    # e.g., capacity fade: we measure capacity at end of each discharge.
    capacity_fade = []
    cycles = list(range(1, N_cycles+1))
    for i in range(N_cycles):
        # PyBaMM provides capacity outputs in the solution (if experiment includes reference discharge).
        # We can compute discharged capacity from the first step of each cycle:
        Q = sol.cycles[i]["Capacity [A.h]"].data[-1] if "Capacity [A.h]" in sol.cycles[i] else None
        if Q is None:
            # If not directly available, compute from current integral over discharge step:
            discharge_step = sol.cycles[i].steps[0]  # first step is discharge
            Q = discharge_step["Discharge capacity [A.h]"].data[-1]
        capacity_fade.append(Q)
    
    # Plot if requested
    results = {"solution": sol, "capacity_fade": capacity_fade, "cycles": cycles}
    if plot:
        # Plot voltage profile of the last cycle vs the first cycle
        import matplotlib.pyplot as plt
        t = sol["Time [h]"].entries
        V = sol["Voltage [V]"].entries
        plt.figure(figsize=(6,4))
        plt.plot(t, V, label="Voltage profile")
        plt.xlabel("Time [h]")
        plt.ylabel("Voltage [V]")
        plt.title("Cell Voltage vs Time")
        plt.legend()
        plt.show()
        
        # Plot capacity fade
        plt.figure(figsize=(5,4))
        plt.plot(cycles, [c*1000 for c in capacity_fade], 'o-')  # convert Ah to mAh for readability
        plt.xlabel("Cycle number")
        plt.ylabel("Discharge Capacity [mAh]")
        plt.title("Capacity Fade over Cycling")
        plt.grid(True)
        plt.show()
    return results

# Example usage:
param_values = load_parameters("battery_params.yaml")
results = simulate_battery(param_values, N_cycles=5, cycle_current=1.0, plot=True)
