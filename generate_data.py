import pandas as pd
import random

# Define possible values and probabilities
road_widths = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11.5, 12, 13, 14, 15]  # Some widths as non-whole numbers
road_width_probs = [0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.07, 0.1, 0.15, 0.15, 0.1, 0.08, 0.06, 0.03, 0.04, 0.03, 0.03, 0.03]  # Adjusted probabilities

smoothness_levels = ["good", "excellent", "intermediate", "bad", "very_bad"]
smoothness_probs = [0.4, 0.2, 0.2, 0.1, 0.1]  # Higher chance for good and excellent

surface_types = ["paved", "asphalt", "chipseal", "concrete", "paving_stones", "bricks"]
surface_probs = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]  # More common surfaces have higher chances

slope_levels = ["flat", "gentle", "moderate", "steep"]  # Common slopes
slope_probs = [0.5, 0.3, 0.15, 0.05]  # Flat is more common

bike_lane_probs = [0.7, 0.3]  # Higher chance of no bike lane

# Load the CSV file
df = pd.read_csv("user-case1-distance-only.csv")

# Add random columns
df["road_width"] = random.choices(road_widths, weights=road_width_probs, k=len(df))
df["smoothness"] = random.choices(smoothness_levels, weights=smoothness_probs, k=len(df))
df["surface"] = random.choices(surface_types, weights=surface_probs, k=len(df))
df["slope"] = random.choices(slope_levels, weights=slope_probs, k=len(df))
df["bike_lane"] = random.choices([0, 1], weights=bike_lane_probs, k=len(df))

# Save the modified CSV
output_path = "updated_data.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
