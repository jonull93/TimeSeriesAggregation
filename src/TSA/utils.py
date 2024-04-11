import sys
import os
sys.path.append(r"./")

baseload = ['U', "Nuclear", 'B', 'H', 'W', "Baseload"]
baseload = baseload + [i.lower() for i in baseload]
CCS = ['BCCS', 'HCCS', 'GCCS', 'BWCCS', 'BECCS', 'HWCCS', 'WGCCS', 'GWGCCS', ]
CHP = ['WA_CHP', 'B_CHP', 'W_CHP', 'H_CHP', 'G_CHP', 'WG_CHP', "CHP"]
midload = ['G', 'WG', 'WG_PS',]
peak = ['G_peak', 'WG_peak', "Peak"]
thermals = baseload + CCS + CHP + midload + ["Other thermals"] + peak + ["Fossil thermals", "Bio thermals", "Thermals"]
nonPeak_thermals = list(thermals)
nonPeak_thermals.remove("WG_peak")
nonPeak_thermals.remove("Peak")
wind = ["Wind", 'WOFF', 'WON', 'wind_onshore', 'wind_offshore', ] + ["WON" + ab + str(dig) for ab in ["A", "B"] for dig
                                                                     in range(5, 0, -1)] + ["WOFF" + str(dig) for dig in range(5, 0, -1)]
PV = ["PVPA1", "PVPA2", "PVPA3", "PVPA4", "PVPA5", "PVR1", "PVR2", "PVR3", "PVR4", "PVR5", "PV", "Solar PV"]
VRE = wind + PV
hydro = ["RO", "RR", "Hydro"]
H2 = ['electrolyser', 'H2store', 'FC', "H2"]
PtH = ['HP', 'EB', 'PtH']
bat = ['bat', 'bat_cap', "Battery", "Bat. power", "Bat. energy"]
PS = ["flywheel", "sync_cond", "super_cap"]

tech_names = []
if os.path.exists(r"data/input/additional_tech_names.txt"):
    with open(r"data/input/additional_tech_names.txt") as f:
        for line in f:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            tech_names += [i.strip() for i in line.split(";")]

tech_names += hydro + thermals + ["bat_cap", "Bat. power", "FC"] + VRE + H2 + bat + PS + PtH
# now remove duplicates
tech_names = list(dict.fromkeys(tech_names))

if __name__ == "__main__":
    print(tech_names)