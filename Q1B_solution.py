import pulp
import math
import os

# ==========================================
# 1. Core Input Data Extraction
# ==========================================

quarters = ['Q1_26', 'Q2_26', 'Q3_26', 'Q4_26', 'Q1_27', 'Q2_27', 'Q3_27', 'Q4_27']
fabs = ['Fab 1', 'Fab 2', 'Fab 3']
techs = ['Mintech', 'TOR']  # Differentiate technologies to avoid mixed RPT calculations
base_ws = ['A', 'B', 'C', 'D', 'E', 'F']

# Fab physical space limits (m^2)
space_limits = {'Fab 1': 1465, 'Fab 2': 1265, 'Fab 3': 665}

# Demand data: demand[Node][Quarter_index]
demand = {
    'Node 1': [12000, 10000, 8500, 7500, 6000, 5000, 4000, 2000],
    'Node 2': [5000, 5200, 5400, 5600, 6000, 6500, 7000, 7500],
    'Node 3': [3000, 4500, 7000, 8000, 9000, 11000, 13000, 16000]
}

# Tool specifications (CapEx in USD, Area in m^2, Util as decimal)
# Dictionary key format: "{Workstation}_{Tech}"
tool_specs = {
    'A_Mintech': {'capex': 4500000, 'area': 6.78, 'util': 0.78},
    'A_TOR':     {'capex': 6000000, 'area': 6.93, 'util': 0.84},
    'B_Mintech': {'capex': 6000000, 'area': 3.96, 'util': 0.76},
    'B_TOR':     {'capex': 8000000, 'area': 3.72, 'util': 0.81},
    'C_Mintech': {'capex': 2200000, 'area': 5.82, 'util': 0.80},
    'C_TOR':     {'capex': 3200000, 'area': 5.75, 'util': 0.86},
    'D_Mintech': {'capex': 4000000, 'area': 5.61, 'util': 0.80},
    'D_TOR':     {'capex': 5500000, 'area': 5.74, 'util': 0.88},
    'E_Mintech': {'capex': 3500000, 'area': 4.65, 'util': 0.76},
    'E_TOR':     {'capex': 5800000, 'area': 4.80, 'util': 0.84},
    'F_Mintech': {'capex': 6000000, 'area': 3.68, 'util': 0.80},
    'F_TOR':     {'capex': 8000000, 'area': 3.57, 'util': 0.90}
}

# Q1'26 Initial tool count dictionary
initial_tools = {
    'Fab 1': {'A_Mintech':50, 'B_Mintech':25, 'C_Mintech':0,  'D_Mintech':50, 'E_Mintech':40, 'F_Mintech':90},
    'Fab 2': {'A_Mintech':35, 'B_Mintech':30, 'C_Mintech':0,  'D_Mintech':50, 'E_Mintech':30, 'F_Mintech':60},
    'Fab 3': {'A_Mintech':0,  'B_Mintech':0,  'C_Mintech':40, 'D_Mintech':35, 'E_Mintech':16, 'F_Mintech':36}
}

# Initialize all TOR tools to 0 for the base period
for f in fabs:
    for ws in base_ws:
        initial_tools[f][f'{ws}_TOR'] = 0

# Process steps and RPT dictionary per Node (Step index: (Required WS, Mintech_RPT, TOR_RPT))
steps_data = {
    'Node 1': {
        1:('D',14,12), 2:('F',25,21), 3:('F',27,23), 4:('A',20,16), 5:('F',12,9),
        6:('D',27,21), 7:('D',17,13), 8:('A',18,16), 9:('A',16,13), 10:('D',14,11), 11:('F',18,16)
    },
    'Node 2': {
        1:('F',19,16), 2:('B',20,18), 3:('E',10,7), 4:('B',25,19), 5:('B',15,11),
        6:('F',16,14), 7:('F',17,15), 8:('B',22,16), 9:('E',7,6), 10:('E',9,7),
        11:('E',20,19), 12:('F',21,18), 13:('E',12,9), 14:('E',15,12), 15:('E',13,10)
    },
    'Node 3': {
        1:('C',21,20), 2:('D',9,7), 3:('E',24,23), 4:('E',15,11), 5:('F',16,14),
        6:('D',12,11), 7:('C',24,21), 8:('C',19,13), 9:('D',15,13), 10:('D',24,20),
        11:('E',17,15), 12:('E',18,13), 13:('F',20,18), 14:('C',12,11), 15:('D',11,10),
        16:('C',25,20), 17:('F',14,13)
    }
}

UPTIME_MINS = 7 * 24 * 60  # Total minutes per week
MOVEOUT_COST = 1000000     # Moveout cost
TRANSFER_COST = 50 * 13    # Quarterly OpEx per wafer for cross-fab transfer

# ==========================================
# 2. Linear Programming Model Setup
# ==========================================

model = pulp.LpProblem("Capacity_Optimization", pulp.LpMinimize)

# Decision Variables
# 1. Loading: Allocated per quarter, node, step, fab, and technology (for accurate RPT)
Load = {}
for q in quarters:
    for n in steps_data.keys():
        for s in steps_data[n].keys():
            for f in fabs:
                for t in techs:
                    Load[(q, n, s, f, t)] = pulp.LpVariable(f"Load_{q}_{n}_S{s}_{f}_{t}", lowBound=0)

# 2. Tools (Total), Purchase (New additions), Moveout (Removals) - Continuous for LP relaxation
Tools = {}
Purch = {}
Move  = {}
for q in quarters:
    for f in fabs:
        for ws in base_ws:
            for t in techs:
                tool_key = f"{ws}_{t}"
                Tools[(q, f, tool_key)] = pulp.LpVariable(f"T_{q}_{f}_{tool_key}", lowBound=0)
                Purch[(q, f, tool_key)] = pulp.LpVariable(f"P_{q}_{f}_{tool_key}", lowBound=0)
                Move[(q, f, tool_key)]  = pulp.LpVariable(f"M_{q}_{f}_{tool_key}", lowBound=0)

# 3. Transfer (Cross-fab volume) - Auxiliary variable for linear formulation without conditional logic
Trans = {}
for q in quarters:
    for n in steps_data.keys():
        for s in list(steps_data[n].keys())[1:]: # Calculate transfers starting from Step 2
            for f in fabs:
                Trans[(q, n, s, f)] = pulp.LpVariable(f"Trans_{q}_{n}_S{s}_{f}", lowBound=0)

# ==========================================
# 3. Objective Function (Minimize Total Cost)
# ==========================================

total_capex = pulp.lpSum(Purch[(q, f, tk)] * tool_specs[tk]['capex'] for q in quarters for f in fabs for tk in tool_specs.keys())
total_moveout = pulp.lpSum(Move[(q, f, tk)] * MOVEOUT_COST for q in quarters for f in fabs for tk in tool_specs.keys())
total_transfer = pulp.lpSum(Trans[(q, n, s, f)] * TRANSFER_COST for q in quarters for n in steps_data.keys() for s in list(steps_data[n].keys())[1:] for f in fabs)

model += total_capex + total_moveout + total_transfer, "Total_Cost"

# ==========================================
# 4. Constraints
# ==========================================

# (A) Demand Fulfillment Constraint (Sum of Mintech and TOR loading = Demand)
for q_idx, q in enumerate(quarters):
    for n in steps_data.keys():
        target_demand = demand[n][q_idx]
        for s in steps_data[n].keys():
            model += pulp.lpSum(Load[(q, n, s, f, t)] for f in fabs for t in techs) == target_demand, f"Demand_{q}_{n}_S{s}"

# (B) Cross-Fab Transfer Constraint (Outflow volume)
for q in quarters:
    for n in steps_data.keys():
        for s in list(steps_data[n].keys())[1:]:
            for f in fabs:
                # Transfer >= Previous step loading in current fab - Current step loading in current fab
                load_prev_step = pulp.lpSum(Load[(q, n, s-1, f, t)] for t in techs)
                load_curr_step = pulp.lpSum(Load[(q, n, s, f, t)] for t in techs)
                model += Trans[(q, n, s, f)] >= load_prev_step - load_curr_step, f"CalcTrans_{q}_{n}_S{s}_{f}"

# (C) Inter-Quarter Tool Balance Constraint (T_q = T_q-1 + Purchase - Moveout)
for q_idx, q in enumerate(quarters):
    for f in fabs:
        for tk in tool_specs.keys():
            if q_idx == 0:
                # Q1 is the initial state: fixed tools, no purchases, no moveouts permitted
                model += Tools[(q, f, tk)] == initial_tools[f][tk], f"InitTool_{f}_{tk}"
                model += Purch[(q, f, tk)] == 0, f"NoPurch_Q1_{f}_{tk}"
                model += Move[(q, f, tk)] == 0, f"NoMove_Q1_{f}_{tk}"
            else:
                prev_q = quarters[q_idx - 1]
                model += Tools[(q, f, tk)] == Tools[(prev_q, f, tk)] + Purch[(q, f, tk)] - Move[(q, f, tk)], f"ToolBal_{q}_{f}_{tk}"

# (D) Physical Space Limits
for q in quarters:
    for f in fabs:
        used_space = pulp.lpSum(Tools[(q, f, tk)] * tool_specs[tk]['area'] for tk in tool_specs.keys())
        model += used_space <= space_limits[f], f"Space_{q}_{f}"

# (E) Tool Capacity Constraints (Capacity >= Requirement)
for q in quarters:
    for f in fabs:
        for ws in base_ws:
            for tech in techs:
                tool_key = f"{ws}_{tech}"
                required_tools = 0
                for n in steps_data.keys():
                    for s, step_info in steps_data[n].items():
                        req_ws = step_info[0]
                        if req_ws == ws:
                            rpt = step_info[1] if tech == 'Mintech' else step_info[2]
                            util = tool_specs[tool_key]['util']
                            capacity_per_tool = UPTIME_MINS * util
                            required_tools += Load[(q, n, s, f, tech)] * rpt / capacity_per_tool
                
                # Total capacity requirement must be <= total available tools in the fab
                model += required_tools <= Tools[(q, f, tool_key)], f"Cap_{q}_{f}_{tool_key}"

# ==========================================
# 5. Model Execution
# ==========================================
print("Model constructed. Executing solver for linear relaxation...")
model.solve(pulp.PULP_CBC_CMD(msg=False))

print("="*40)
print(f"Solution Status: {pulp.LpStatus[model.status]}")
print(f"Global Optimal Total Cost: ${pulp.value(model.objective):,.2f}")
print("="*40)

if model.status != pulp.LpStatusOptimal:
    print("Model is infeasible. Please check input data and space constraints.")
    exit()

# Print Cost Breakdown and Action Recommendations
total_capex_val = sum(Purch[(q, f, tk)].varValue * tool_specs[tk]['capex'] for q in quarters for f in fabs for tk in tool_specs.keys())
total_moveout_val = sum(Move[(q, f, tk)].varValue * MOVEOUT_COST for q in quarters for f in fabs for tk in tool_specs.keys())
total_transfer_val = sum(Trans[(q, n, s, f)].varValue * TRANSFER_COST for q in quarters for n in steps_data.keys() for s in list(steps_data[n].keys())[1:] for f in fabs)

print("\n[Cost Breakdown]")
print(f"  - New Tool Purchase (CapEx): ${total_capex_val:,.2f}")
print(f"  - Old Tool Moveout (OpEx):   ${total_moveout_val:,.2f}")
print(f"  - Cross-Fab Transfer (OpEx): ${total_transfer_val:,.2f}\n")

print("[Key Actions: Tool Purchases and Moveouts (Non-zero only)]")
for q in quarters:
    print(f"--- {q} ---")
    for f in fabs:
        for tk in tool_specs.keys():
            p_val = Purch[(q, f, tk)].varValue
            m_val = Move[(q, f, tk)].varValue
            if p_val > 0:
                print(f"  {f} Purchase {int(p_val)} units of {tk}")
            if m_val > 0:
                print(f"  {f} Moveout {int(m_val)} units of {tk}")

# ==========================================
# 6. Heuristic Post-Processing I: Step-level Integer Allocation
# ==========================================
final_loading = {}
for q in quarters:
    q_idx = quarters.index(q)
    for n in ['Node 1', 'Node 2', 'Node 3']:
        demand_val = demand[n][q_idx]  # Total demand for the quarter
        
        # Ensure 100% integer reconciliation independently for each step
        for s in steps_data[n].keys():
            loads = []
            for f in fabs:
                # Combine continuous solutions for Mintech and TOR
                val = Load[(q, n, s, f, 'Mintech')].varValue or 0.0
                val += Load[(q, n, s, f, 'TOR')].varValue or 0.0
                loads.append({'f': f, 'val': val, 'int_val': int(val)})
            
            # Use the Largest Remainder Method to ensure exact reconciliation with demand
            current_sum = sum(item['int_val'] for item in loads)
            deficit = int(round(demand_val - current_sum))
            
            # Sort by remainder in descending order
            loads.sort(key=lambda x: x['val'] - x['int_val'], reverse=True)
            
            # Allocate the deficit starting with the largest remainder
            for i in range(deficit):
                loads[i]['int_val'] += 1
                
            # Store in final dictionary
            for item in loads:
                final_loading[(q, n, s, item['f'])] = item['int_val']

# ==========================================
# 7. Heuristic Post-Processing II: Excel Waterfall Simulation
# ==========================================
final_tools = {}
for q_idx, q in enumerate(quarters):
    for f in fabs:
        for ws in base_ws:
            mintech_tk = f"{ws}_Mintech"
            tor_tk = f"{ws}_TOR"
            
            # 1. Determine Mintech tool count
            if q_idx == 0:
                mintech_avail = initial_tools[f][mintech_tk]
            else:
                # Adopt LP's Mintech retention strategy with standard rounding
                mintech_avail = int(round(Tools[(q, f, mintech_tk)].varValue or 0.0))
            final_tools[(q, f, mintech_tk)] = mintech_avail
            
            # 2. Simulate Excel's row-by-row overflow algorithm to calculate TOR requirements
            cumul_mintech_req = 0.0
            total_tor_req = 0.0
            
            # Traverse strictly according to Excel row sequence (Node 1 -> 2 -> 3)
            for n in ['Node 1', 'Node 2', 'Node 3']:
                for s in sorted(steps_data[n].keys()):
                    step_info = steps_data[n][s]
                    if step_info[0] == ws:
                        load = final_loading[(q, n, s, f)]
                        
                        min_rpt = step_info[1]
                        tor_rpt = step_info[2]
                        min_cap = UPTIME_MINS * tool_specs[mintech_tk]['util']
                        tor_cap = UPTIME_MINS * tool_specs[tor_tk]['util']
                        
                        min_req = load * min_rpt / min_cap
                        tor_req = load * tor_rpt / tor_cap
                        
                        previous_cumul = cumul_mintech_req
                        cumul_mintech_req += min_req
                        
                        # Calculate capacity overflow
                        if previous_cumul >= mintech_avail:
                            row_overflow = min_req
                        elif cumul_mintech_req > mintech_avail:
                            row_overflow = cumul_mintech_req - mintech_avail
                        else:
                            row_overflow = 0.0
                            
                        # Convert overflow to TOR requirements and accumulate
                        if min_req > 0:
                            req_on_tor = row_overflow * (tor_req / min_req)
                            total_tor_req += req_on_tor
            
            # 3. Determine TOR tool count
            lp_tor = int(round(Tools[(q, f, tor_tk)].varValue or 0.0))
            if q_idx == 0:
                final_tools[(q, f, tor_tk)] = initial_tools[f][tor_tk]
            else:
                # Take the maximum of simulated requirements and LP suggestions, then round up
                final_tools[(q, f, tor_tk)] = int(math.ceil(max(total_tor_req, lp_tor)))

# ==========================================
# 8. Output Configuration: Tooling Matrix
# ==========================================
print("\n[Tooling Plan Matrix]")
for q in quarters:
    print(f"\n--- {q} ---")
    # Print Mintech (First 6 rows)
    for ws in base_ws:
        print(f"{final_tools[(q, 'Fab 1', ws+'_Mintech')]}\t{final_tools[(q, 'Fab 2', ws+'_Mintech')]}\t{final_tools[(q, 'Fab 3', ws+'_Mintech')]}")
    # Print TOR (Last 6 rows)
    for ws in base_ws:
        print(f"{final_tools[(q, 'Fab 1', ws+'_TOR')]}\t{final_tools[(q, 'Fab 2', ws+'_TOR')]}\t{final_tools[(q, 'Fab 3', ws+'_TOR')]}")

# ==========================================
# 9. Output Configuration: Loading Matrix Export
# ==========================================
output_file = "loading_results.txt"
with open(output_file, "w") as f_out:
    for q in quarters:
        for n in ['Node 1', 'Node 2', 'Node 3']:
            for s in sorted(steps_data[n].keys()):
                for f in fabs:
                    f_out.write(f"{final_loading[(q, n, s, f)]}\n")

print(f"\nLoading data successfully exported to '{output_file}'.")
print(f"Path: {os.path.join(os.getcwd(), output_file)}")
print("Please copy the contents of this text file and paste it directly into the 'Loading (to fill)' column in the Excel template.")