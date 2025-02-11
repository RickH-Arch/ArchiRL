import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd


class ColorScaleManager:

    _instance = None

    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            cls._instance = super(ColorScaleManager, cls).__new__(cls,*args,**kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.color_dict = {'agent_color':"rgb(178,19,9)",#智能体当前所处位置颜色
                           'undefined_color': "rgb(246,248,234)",
                           'entrance_color' : "rgb(156,155,151)",
                           'path_color' : "rgb(241,199,135)",
                           'park_color' : "rgb(86,145,170)"}
        self.key_list = list(self.color_dict.keys())

    
    def getColorValue(self,color_name):
        '''根据颜色名称返回0-1范围内对应颜色的起始值、中间值和结束值'''
        
        if color_name not in self.key_list:
            return (0,0,0)
        
        index = self.key_list.index(color_name)
        
        return (index/len(self.color_dict),
                (index + 0.5)/len(self.color_dict),
                (index + 1)/len(self.color_dict))

    def getColorScale(self):
        '''返回plotly读取的离散色卡'''
        scale = []
        for i,n in enumerate(self.key_list):
            value = self.getColorValue(n)
            scale.append([value[0],self.color_dict[n]])
            scale.append([value[-1],self.color_dict[n]])
        
        return scale
        
csMangr = ColorScaleManager()

def showGrid(grid,width = 480,height = 450,title = "Parking Grid"):
    g = np.array(grid)
    g = g[~np.isnan(g)]
    fig = go.Figure(data=go.Heatmap(
                    z=grid,
                    zmin = 0,
                    zmax=1,
                    colorscale = csMangr.getColorScale(),
                    ))

    fig.update_layout(
        title=title,
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, b=50, t=50),
        showlegend = False,
    )
    fig.show()
    # fig = px.density_heatmap(grid,x = "x",y = "y",
    #                          title=title,
    #                          )

    # fig.update_layout(
    #     autosize=False,
    #     width=width,
    #     height=height,
    #     margin=dict(l=20, r=20, b=50, t=50),
    #     showlegend=False,
    #     coloraxis_showscale = False,
        
    # )
    # fig.show()