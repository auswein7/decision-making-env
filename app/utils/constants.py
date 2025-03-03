JSON_LOAD_PATH = "scenarios/test_scenario.json" # file path to preload system configurations
JSON_SAVE_PATH = "app/out" # save directory for all experiment data
BEST_RESPONSE="best_response"
APPROX_BEST_RESPONSE = "approximate_best_response"
LOGIT_RESPONSE = "logit_response"
GENETIC_RESPONSE = "genetic_response"
BRUTE_FORCE = "brute_force"
BETA = "beta"
TEMP = "temperature"
MAX_BETA = 1.0 # maximum beta value when conducting beta analysis
BETA_STEP_SIZE = 0.05 # amount to change beta by each trial
# TODO:: may need to adjust maximum and step size for temp, good info may have temp bound [1, 10]
MAX_TEMP = 50 # maximum temperature value when conducting temperature analysis
TEMP_STEP_SIZE = 10 # amount to change temperature by each trial
HIST_BINS = 10 # amount of bins for histogram generation