from ultralytics import YOLO
import cvzone
import matplotlib.pyplot as plt

import yaml
from typing import List

from constants import classNames

def make_yaml(train_path:str, val_path:str, nc:int, names_array: List[str]) -> None:
    data_yaml = dict(
    train = train_path,
    val = val_path,
    nc = nc,
    names = names_array
    )
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)


### Ploting

# TODO: Improve this function
def plot_vehicle_pie_chart(vehicles_dict, plot_title = "Modal Split"):
    """
    Plot a pie chart of vehicle types from the input dictionary.

    Args:
        vehicles_dict (dict): Dictionary containing vehicle types as keys and instance counts as values.

    Returns:
        None
    """
    labels = [classNames[key] for key in vehicles_dict.keys()]
    sizes = list(vehicles_dict.values())  # Get the instance counts

    plt.pie(sizes, labels=labels, autopct='%1.1f%%')  # Create the pie chart
    plt.title(plot_title)  # Add a title
    plt.show()  # Display the plot
