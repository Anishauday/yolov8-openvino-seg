import pandas as pd
import numpy as np
from IPython import display
import os, cv2
import matplotlib.pyplot as plt
import gradio as gr

# Globals
results_data = pd.DataFrame(columns=['model', 'precision', 'device', 'goal', 'latency', 'throughput', 'marker', 'color'])
face_detect_dir = "models/face_detection"
mobilenet_dir = "models/mobilenet-v2"
unet_dir = "models/unet-camvid"
reports_dir = "."

model_dict = {
        "face_detect-FP32": ('o', 'tab:blue', f"{face_detect_dir}/FP32/face-detection-adas-0001.xml"),
        "face_detect-FP16": ('s', 'tab:orange', f"{face_detect_dir}/FP16/face-detection-adas-0001.xml"),
        "face_detect-INT8": ('^', 'tab:green', f"{face_detect_dir}/FP16-INT8/face-detection-adas-0001.xml"),
        "mobilenet-v2-FP32": ('*', 'tab:red', f"{mobilenet_dir}/FP32/mobilenet-v2-pytorch.xml"),
        "mobilenet-v2-FP16": ('P', 'tab:purple', f"{mobilenet_dir}/FP16/mobilenet-v2-pytorch.xml"),
        "mobilenet-v2-INT8": ('d', 'tab:brown', f"{mobilenet_dir}/INT8/quantized_mobilenet_v2.xml"),
        "unet-camvid-FP32": ('p', 'tab:pink', f"{unet_dir}/FP32/unet-camvid-onnx-0001.xml"),
        "unet-camvid-FP16": ('v', 'tab:olive', f"{unet_dir}/FP16/unet-camvid-onnx-0001.xml"),
        "unet-camvid-INT8": ('>', 'tab:cyan', f"{unet_dir}/FP16-INT8/unet-camvid-onnx-0001.xml"),
}

def build_gr_blocks(available_devices, goal):
    model_options = ['face_detect', 'mobilenet-v2', 'unet-camvid']
    precision_options = ['FP32', 'FP16', 'INT8']
    if goal == "Throughput":
        device_options = available_devices + ["AUTO throughput"]
    else:
        device_options = available_devices + ["AUTO"]

    with gr.Blocks(fill_width=True) as demo:
        gr.Markdown(f"# Benchmark Model {goal} by Device")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                model_name = gr.Dropdown(model_options, label="Choose a model", value=model_options[0])
                precision = gr.Dropdown(precision_options, label="Choose a precision", value=precision_options[0])
                device = gr.Dropdown(device_options, label="Choose a device", value=device_options[0])
                hint = gr.Radio(choices=goal.lower(), label="Goal", value=goal.lower(), visible=False)
                run = gr.Button("Run Benchmark", variant="primary")
                clear = gr.Button("Clear Plot Data")
            with gr.Column(scale=2, min_width=300):
                plot = gr.Plot()
        with gr.Row():
            cmd = gr.Textbox(label="Command used for this run:")
            run.click(run_benchmark, inputs=[model_name, precision, device, hint], outputs=[plot, cmd])
            clear.click(clear_plot_results, inputs=[model_name, hint], outputs=[plot])
    return demo

def resize_image(img_in, layer_shape):

    height,width=img_in.shape[0:2]

    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = layer_shape

    # Resize, Reshape the image to meet network input sizes.
    resized_image = cv2.resize(img_in, (W, H))
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    img_out = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    return img_out, height, width

def draw(image,boxes):
    new_image=image.copy()
    color = (0,200,0)
    for box in boxes:
        x1,y1,x2,y2=box
        cv2.rectangle(img=new_image, pt1=(x1,y1), pt2=(x2, y2), color=color, thickness=10)
    return new_image
    
def parse_output(file_path, goal):
    import numpy as np
    
    median_latency = np.nan
    throughput = np.nan

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:     
            for line in file:
                if line.startswith("[ INFO ]    Median:"):
                    parts = line.split()
                    value_str, unit = parts[-2], parts[-1]
                    median_latency = float(value_str)
                if line.startswith("[ INFO ] Throughput:"):
                    parts = line.split()
                    value_str, unit = parts[-2], parts[-1]
                    throughput = float(value_str)
    else:
        raise Exception(f"{file_path} could not be found.")

    # Check if the value was found
    if goal == "latency" and median_latency == np.nan:
        raise Exception("The line with the median latency was not found in the file.")
        
    if goal == "throughput" and throughput == np.nan:
        raise Exception("The line with the throughput was not found in the file.")    

    return median_latency, throughput

def update_plot(model_name, precision, device, goal, results):  
    import matplotlib.pyplot as plt
    
    # Globals
    global results_data, model_dict

    key_name = f"{model_name}-{precision}"
    marker, color, _ = model_dict[key_name]
    new_row = pd.DataFrame({'model': [model_name], 
                            'precision': [precision],
                            'device': [device], 
                            'goal': [goal], 
                            'latency': [results[0]],
                            'throughput': [results[1]],
                            'marker': marker,
                            'color': color,
                           })
    results_data = pd.concat([results_data, new_row], ignore_index=True)
    # print(results_data)
    fig, ax = plt.subplots()

    # Loop through each row. If that row has a goal equal to the goal we're plotting, then plot its corresponding results
    for index, row in results_data.iterrows():
        # Only plotting either latency or throughput
        if row['goal'] == goal:
            # Only plotting one model at a time
            if row['model'] == model_name:
                plt.scatter(row['device'], row[goal], marker=row['marker'], color=row['color'])
    
    # Create custom legend with only the unique models in results_data
    results_models = results_data.drop_duplicates(subset = ['model','precision','goal'])
    for index, row in results_models.iterrows():
        if row['goal'] == goal:
            if row['model'] == model_name:
                plt.scatter([], [], label=row['precision'], marker=row['marker'], color=row['color'])

    # Set the x-axis labels and title
    ax.set_xlabel('Device')
    if goal == "latency":
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_name} Latency by Device")
    else:
        ax.set_ylabel("Throughput (FPS)")
        ax.set_title(f"{model_name} Throughput by Device")
    
    ax.legend(loc='best')
    return fig

def run_benchmark(model_name, precision, device, hint):

    global model_dict, reports_dir

    key_name = f"{model_name}-{precision}"
    _,_,model_path = model_dict.get(key_name)

    if device == "AUTO throughput":
        bm_cmd_hint = "cumulative_throughput"
        # If an NPU is present, it won't be included in AUTO (as of 2024.3). So let's manually specify all the available devices.
        device = "AUTO:GPU,NPU,CPU"
    else:
        bm_cmd_hint = hint
        # and leave device unchanged

    # File to save the benchmark_app output to
    file_path = f"{reports_dir}/{model_name}-{precision}-{device}-{hint}.txt"
    file_path = file_path.replace(',','-')
    file_path = file_path.replace(':','-')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    if precision == "INT8":
        if device == "NPU":
            precision_arg = "-infer_precision i8"
        else:
            precision_arg = "-infer_precision int8"
    elif precision == "FP16":
        precision_arg = "-infer_precision f16"
    else:
        precision_arg = "-infer_precision f32"
    
    bm_command = (f"benchmark_app -m {model_path} -hint {bm_cmd_hint} -d {device} {precision_arg} -t 30")
    os.system(f"{bm_command} 1> {file_path}")
    
    results = parse_output(file_path, hint)

    fig = update_plot(model_name, precision, device, hint, results)
    return fig, bm_command

def clear_plot_results(model_name, goal):
    global results_data

    # Clear the dataframe
    results_data = pd.DataFrame(columns=['model', 'precision', 'device', 'goal', 'latency', 'throughput', 'marker', 'color'])

    # Redraw the empty plot
    fig, ax = plt.subplots()
    
        # Set the x-axis labels and title
    ax.set_xlabel('Device')
    if goal == "latency":
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_name} Latency by Device")
    else:
        ax.set_ylabel("Throughput (FPS)")
        ax.set_title(f"{model_name} Throughput by Device")

    ax.legend(loc='best')
    return fig