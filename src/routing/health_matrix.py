import yaml

def load_health_matrix():
    with open("configs/routing.yaml", "r") as f:
        return yaml.safe_load(f)["routing"]["health_profiles"]

def get_sensitivity(profile_name: str) -> float:
    profiles = load_health_matrix()
    if profile_name in profiles:
        return profiles[profile_name]["sensitivity"]
    return 1.0

def get_beta_aqi(profile_name: str) -> float:
    profiles = load_health_matrix()
    if profile_name in profiles:
        return profiles[profile_name]["beta_aqi"]
    return 0.1

def get_met(transport_mode: str) -> float:
    with open("configs/routing.yaml", "r") as f:
        modes = yaml.safe_load(f)["routing"].get("transport_modes", {})
    if transport_mode in modes:
        return modes[transport_mode].get("met", 1.5)
    return 1.5
