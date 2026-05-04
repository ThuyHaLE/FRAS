# features/target.py

# Anonymized fire event identifier (stable random remap, no temporal meaning)
TARGET_EVENT_ID = "event_id" 

# Event indicator, 1 if fire hit within 72h, 0 if censored (never hit)
TARGET_EVENT = "event"

# Time from t0+5h until fire comes within 5 km of an evac zone (hours). 
# For censored events (never hit within 72h), this is the last observed time within the window (<= 72).
TARGET_TIME = "time_to_hit_hours"