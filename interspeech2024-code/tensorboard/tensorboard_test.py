# Cynthia Chen 11/6/2025
# Purpose: Test reading TensorBoard event files, determine whether the logs are empty

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Base path is the directory of this script
base = Path(__file__).resolve().parent  # tensorboard folder

# Convert Path to string
event_path = str(base / 'stutternet_en' / 'events.out.tfevents.1762490482.CRC-Laptop')

ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

print(ea.Tags())
