import taipy as tp
from taipy.gui import Gui, State
import pandas as pd

scenarios = tp.get_scenarios()
scenario_names = [scenario.name for scenario in scenarios]
scenario_results = [scenario.results.read() for scenario in scenarios]
scenario_output_paths = [scenario.output_path.read() for scenario in scenarios]

selected_metric = "Validation mAP 0.5"
metrics_list = [
    "Box Loss",
    "Obj Loss",
    "Cls Loss",
    "Precision",
    "Recall",
    "Validation mAP 0.5",
    "Validation mAP 0.5:0.95",
]
metrics_dict = {
    "Epoch": "               epoch",
    "Box Loss": "      train/box_loss",
    "Obj Loss": "      train/obj_loss",
    "Cls Loss": "      train/cls_loss",
    "Precision": "   metrics/precision",
    "Recall": "      metrics/recall",
    "Validation mAP 0.5": "     metrics/mAP_0.5",
    "Validation mAP 0.5:0.95": "metrics/mAP_0.5:0.95",
}


def get_chart_data(model_name: str, selected_metric: str) -> list:
    return scenario_results[scenario_names.index(model_name)][
        metrics_dict[selected_metric]
    ].tolist()


chart_data = pd.DataFrame(
    {
        "Epoch": scenario_results[0][metrics_dict["Epoch"]].tolist(),
        "YOLOv5n": get_chart_data("YOLOv5n", selected_metric),
        "YOLOv5s": get_chart_data("YOLOv5s", selected_metric),
        "YOLOv5m": get_chart_data("YOLOv5m", selected_metric),
        "YOLOv5l": get_chart_data("YOLOv5l", selected_metric),
        "YOLOv5x": get_chart_data("YOLOv5x", selected_metric),
    }
)

selected_image = "test_crossroad.png"
image_list = [
    "test_crossroad.png",
    "test_marid_1.png",
    "test_marid_2.png",
    "test_zamak_1.png",
    "test_zamak_2.png",
    "test_complex_1.png",
    "test_complex_2.png",
]
image_path_1 = "runs/detect/gd_v5n_8/test_crossroad.png"
image_path_2 = "runs/detect/gd_v5x_8/test_crossroad.png"
selected_scenario_1 = "YOLOv5n"
selected_scenario_2 = "YOLOv5x"
scenario_list = ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"]

fullscreen_image_1 = False
fullscreen_image_2 = False

models = ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"]
metrics_dict["Epoch"]


def get_best_map(model_name: str) -> float:
    return scenario_results[scenario_names.index(model_name)][
        metrics_dict["Validation mAP 0.5"]
    ].max()


best_map_yolov5n = get_best_map("YOLOv5n")
best_map_yolov5s = get_best_map("YOLOv5s")
best_map_yolov5m = get_best_map("YOLOv5m")
best_map_yolov5l = get_best_map("YOLOv5l")
best_map_yolov5x = get_best_map("YOLOv5x")

latency_data = pd.DataFrame(
    {
        "latency": [6.3, 6.4, 8.2, 10.1, 12.1],
        "YOLOv5n": [best_map_yolov5n, None, None, None, None],
        "YOLOv5s": [None, best_map_yolov5s, None, None, None],
        "YOLOv5m": [None, None, best_map_yolov5m, None, None],
        "YOLOv5l": [None, None, None, best_map_yolov5l, None],
        "YOLOv5x": [None, None, None, None, best_map_yolov5x],
    }
)

latency_layout = {
    "xaxis": {"title": {"text": "Latency (V100 - ms)"}},
    "yaxis": {"title": {"text": "Accuracy (mAP 0.5)"}},
}

latency_marker = {"size": 12, "symbol": "cross"}


def change_metric(state: State):
    state.chart_data = pd.DataFrame(
        {
            "Epoch": scenario_results[0][metrics_dict["Epoch"]].tolist(),
            "YOLOv5n": get_chart_data("YOLOv5n", state.selected_metric),
            "YOLOv5s": get_chart_data("YOLOv5s", state.selected_metric),
            "YOLOv5m": get_chart_data("YOLOv5m", state.selected_metric),
            "YOLOv5l": get_chart_data("YOLOv5l", state.selected_metric),
            "YOLOv5x": get_chart_data("YOLOv5x", state.selected_metric),
        }
    )


def change_image(state: State):
    state.image_path_1 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_1)]}/{state.selected_image}"
    state.image_path_2 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_2)]}/{state.selected_image}"


def change_scenario_1(state: State):
    state.image_path_1 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_1)]}/{state.selected_image}"


def change_scenario_2(state: State):
    state.image_path_2 = f"{scenario_output_paths[scenario_names.index(state.selected_scenario_2)]}/{state.selected_image}"


def fullscreen_1(state: State):
    state.fullscreen_image_1 = not state.fullscreen_image_1


def fullscreen_2(state: State):
    state.fullscreen_image_2 = not state.fullscreen_image_2


page = """
<|container|

# 🛰️ Vehicle Image Recognition

<intro_card|card|

## Comparing the performance of different YOLOv5 models on drone imagery

In this application, we compare the performance of different YOLOv5 models for detecting and recognizing 
military vehicles in drone imagery.

Learn more about this project <a href="https://github.com/AlexandreSajus/Military-Vehicles-Image-Recognition" target="_blank">here</a>.

<br/>

<p align="center">
  <img src="media/example_inference.png" alt="Example Inference" width="40%"/>
</p>

<p align="center">
  <img src="media/model_comparison.png" alt="YOLOv5 Model Sizes" width="50%"/>
</p>


|intro_card>

<br/>

<chart_card|card|

## Metrics over Epochs 📈

<|{selected_metric}|selector|dropdown|lov={metrics_list}|on_change=change_metric|>
<|{chart_data}|chart|type=line|x=Epoch|y[5]=YOLOv5n|y[4]=YOLOv5s|y[3]=YOLOv5m|y[2]=YOLOv5l|y[1]=YOLOv5x|title=Metric over Epochs for YOLOv5 Models|rebuild|>

|chart_card>

<br/>

<pareto_card|card|

## Accuracy vs Latency 📊

<|{latency_data}|chart|mode=markers|x=latency|y[5]=YOLOv5n|y[4]=YOLOv5s|y[3]=YOLOv5m|y[2]=YOLOv5l|y[1]=YOLOv5x|title=Accuracy against latency for YOLOv5 Models|rebuild|layout={latency_layout}|marker={latency_marker}|>

|pareto_card>

<br/>

<image_card|card|

## Compare Results 🖼️

<|{selected_image}|selector|dropdown|lov={image_list}|on_change=change_image|>

<images|layout|columns=1 1|

<|{selected_scenario_1}|selector|dropdown|lov={scenario_list}|on_change=change_scenario_1|class_name=fullwidth|><br/>
<center><|{image_path_1}|image|width=60vh|on_action=fullscreen_1|></center>

<|{selected_scenario_2}|selector|dropdown|lov={scenario_list}|on_change=change_scenario_2|class_name=fullwidth|><br/>
<center><|{image_path_2}|image|width=60vh|on_action=fullscreen_2|></center>

|images>

|image_card>

|>

<|{fullscreen_image_1}|dialog|partial={part_image_1}|on_action={lambda s: s.assign("fullscreen_image_1", False)}|width=150vh|>
<|{fullscreen_image_2}|dialog|partial={part_image_2}|on_action={lambda s: s.assign("fullscreen_image_2", False)}|width=150vh|>
"""

gui = Gui(page)
part_image_1 = gui.add_partial("""<|{image_path_1}|image|width=100%|>""")
part_image_2 = gui.add_partial("""<|{image_path_2}|image|width=100%|>""")
gui.run(dark_mode=False, debug=True, title="🛰️Vehicle Image Recognition")
