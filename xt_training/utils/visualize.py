import os
import shutil

import torch
from xt_training.utils import _import_config
from xt_training import metrics
import base64
from io import BytesIO
import time

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from PIL import Image
import requests

from xt_training.metrics import logit_to_label

from torch.nn import functional as F

def visualize(args):
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
        
    if isinstance(config_path, str) and os.path.isdir(config_path):
        config_dir = config_path
        config_path = os.path.join(config_dir, 'config.py')
        assert checkpoint_path is None, (
            "checkpoint_path is not valid when config_path is a directory.\n"
            "\tSpecify either a config script and (optional) checkpoint individually, or\n"
            "\tspecify a directory containing both config.py and best.pt"
        )
        checkpoint_path = os.path.join(config_dir, 'best.pt')

    if isinstance(config_path, str):
        config = _import_config(config_path)
    else:
        config = config_path

    #  Load model
    model = config.model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = model.to(device)
    if checkpoint_path is not None:
        if os.path.isdir(checkpoint_path):
            checkpoint_path = checkpoint_path + 'best.pt'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transforms = config.valid_transforms
    preprocess_fn = getattr(config, 'preprocess', None)
    postprocess_fn = getattr(config, 'postprocess', None)
    classes = getattr(config, 'classes', None)
    if not classes:
        assert hasattr(config, 'n_classes'), "Must define either classes or n_classes in config"
        classes = range(config.n_classes)

    # colors for visualization
    COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
              '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

    # Start Dash
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div(className='container', children=[
        dbc.Row(html.H1(f"{config_dir}")),

        dbc.Row([
            dbc.Col(
                dbc.Input(id='input-url', placeholder='Insert Image URL...'), width=7
            ),
            dbc.Col(
                dcc.Upload(
                    id="upload-data",
                    children=dbc.Button(
                        "Upload",
                        color="secondary",
                        className='mr-1',
                        outline=True
                    ),
                ),
                width=2
            ),
            dbc.Col([
                dbc.Label("Model Type"),
                dbc.RadioItems(
                    id="model-type",
                    options=[
                        {"label": "Classification", "value": 1},
                        {"label": "Detection", "value": 2},
                    ],
                    value=1
                )
            ])
        ]),
        dbc.Spinner(
            dcc.Graph(id='model-output', style={"height": "70vh"}),
            color="primary"
        )

    ])

    @app.callback(
        Output('model-output', 'figure'),
        [Input('input-url', 'n_submit'),
        Input('upload-data', 'contents')],
        [State('upload-data', 'filename'),
        State('upload-data', 'last_modified'),
        State('input-url', 'value'),
        State('model-type', 'value')])
    def run_model(n_submit, list_of_contents, list_of_names, list_of_dates, url, model_type):
        ctx = dash.callback_context
        triggered_element = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            if not triggered_element:
                return go.Figure().update_layout(title='Grab an image url or upload an image.')
            elif triggered_element == 'upload-data':
                metadata, contents = list_of_contents.split(',')
                buf = BytesIO(base64.b64decode(contents))
                im = Image.open(buf)
                if im.mode == 'RGBA':
                    red, green, blue, mask = im.split()
                    im = Image.merge('RGB', [red, green, blue])
                    # im.convert('RGB')
            else:
                im = Image.open(requests.get(url, stream=True).raw)
        except Exception as e:
            return go.Figure().update_layout(title=f'Error Loading Image: {str(e)}')

        tstart = time.time()
        
        try:
            if model_type == 1:
                # Classification
                torch_im = transforms(im)
                torch_im = torch_im.unsqueeze(0).to(device)
                output = model(torch_im)
                tend = time.time()
                inference_time = tend-tstart
                probs = F.softmax(output, dim=1)
                ypred = metrics.logit_to_label(output)
                pred_class = ypred.item()
                pred_prob = probs.detach().cpu().numpy()[0][pred_class]

                title = f'Prediction: {classes[pred_class]}, {pred_prob*100}% | Inference Time: {inference_time:.2f}s'
                fig = pil_to_fig(im, showlegend=True, title=title)

            elif model_type == 2:
                # Detection
                assert preprocess_fn is not None and postprocess_fn is not None, \
                    "To visualize detection, you must define preprocess and postprocess functions in your config"
                
                torch_im = preprocess_fn(im)
                torch_im = torch_im.to(device)

                output = model(torch_im)
                
                bboxes, labels = postprocess_fn(output[0], classes, torch_im, im)
                tend = time.time()
                title = f'Inference Time: {tend-tstart:.2f}s'
                fig = pil_to_fig(im, showlegend=True, title=title)

                # Get list of bboxes
                existing_classes = set()
                assert len(bboxes) == len(labels)
                for i in range(len(bboxes)):
                    label = labels[i]
                    # confidence = scores[i].max()
                    x0, y0, x1, y1 = bboxes[i]

                    # only display legend when it's not in the existing classes
                    showlegend = label not in existing_classes
                    text = f"class={label}"#<br>confidence={confidence:.3f}"

                    add_bbox(
                        fig, x0, y0, x1, y1,
                        opacity=0.7, group=label, name=label, color=COLORS[classes.index(label)], 
                        showlegend=showlegend, text=text,
                    )

                    existing_classes.add(label)

            else:
                title = f"Unsupported model type: {model_type}"
                fig = pil_to_fig(im, showlegend=True, title=title)
        except Exception as e:
            # Return error as figure title
            return go.Figure().update_layout(title=f'Error: {str(e)}')


        return fig


    app.run_server()





# Plotly Helper Functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))