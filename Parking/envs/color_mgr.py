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