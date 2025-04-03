from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

from kivy.uix.label import Label
from kivy.animation import Animation

from PIL import Image as PImage
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt
import onnxruntime
import torch


import onnx
import helpers as helper





'''

Given an onnx model

- Generate an Input Tensor from sliders
- Produce an Image



Todo
- [ ] Implement Scrolling
- [ ] Backprop to adjust latent dim with visual indication
- [ ] Choice Sample (File Picker)


'''

print(onnxruntime.get_device())



MODEL_NAME = "ThermalImageAutoencoder_50_decoder.onnx"

def LoadModel(model_name):
    onnx_model = onnxruntime.InferenceSession(f"model/{model_name}")
    return onnx_model

def InferModel(input, model):
    out = model.run(None, {
        "input": input
    })
    return np.uint8(out[0].squeeze() * 255)
    

LATENT_DIMS = 50
SIZE_IMAGE = (64, 148)


import random
index_i = 0


class TweakTok(App):
    def on_value_change(self, instance, value):
        index = instance.ids["id"]
        self.input[0][index]=value
        self.labels[instance.ids["id"]].text = f"{index}:{value:0.2f}"
        self.update_texture()

    

    def on_touch_up(self, instance, value):

        pass
    
    

    def update_texture(self):
        clr_data = np.uint8(helper.ironbow_cmap(InferModel(self.input, self.model))[:, :, :3] * 255)
        self.texture.blit_buffer(clr_data.tobytes(), colorfmt="rgb", bufferfmt="ubyte")

    def anims_init(self, animation, slider):
        snext = slider.ids.get("next", None)
        if snext :  
            self.anims[slider.ids["id"] + 1].start(snext)
        else:
            self.random_btn.set_disabled(False)

    def random_foot(self):
        random_samples = np.array([ np.random.randn() * sigma + mu for _, _, mu, sigma in self.mmms ], dtype=np.float32)
        self.anims = []
        for index, sample in enumerate(random_samples):
            sample = min(self.sliders[index].max, float(sample))
            sample = max(self.sliders[index].min, float(sample))
            self.anims += [ Animation(value=sample, duration=0.2) ]
            self.anims[index].bind(on_complete=self.anims_init)
            self.anims[index] &= Animation(value_track_color=[1, 0, 0, 1], value_track=True, duration=0.2)
            if index > 0:
                self.sliders[index - 1].ids["next"] = self.sliders[index]
                

                

        
        self.anims[0].start(self.sliders[0])
        
            
    def random_btn_touched(self, instance, value):
        self.random_foot()
        self.random_btn.disabled_color = [1, 0, 0, 1]
        self.random_btn.set_disabled(True)
        

    def view(self):
        self.root = BoxLayout(orientation="horizontal")

        self.random_btn = Button(text="Random Sample")
        self.random_btn.bind(on_touch_down=self.random_btn_touched)
        
        control = GridLayout(cols=1, spacing=10, size_hint_y=None)
        control.bind(minimum_height=control.setter('height'))


        control.add_widget(self.random_btn)
        
        

        image = Image(size_hint=(1, 1), allow_stretch=True,keep_ratio=True)
        self.texture = Texture.create(size=(64, 148), colorfmt="rgb")
        

        sboxes = [ BoxLayout(orientation="horizontal", size_hint=(1, None)) for _ in range(LATENT_DIMS) ]
        self.sliders = []
        self.labels = []
        sboxes_ren = []
        

        image.texture = self.texture
        for i, sbox in enumerate(sboxes):
            mini, maxi, mean, std = self.mmms[i]
            slider = Slider(min=float(mini), max=float(maxi), value=float(mean), size_hint=(0.7, None))
            label = Label(text=f"{i}:{slider.value:0.2f}", size_hint=(0.3, None))
            
            self.labels += [ label ]
            self.sliders += [ slider ]

            slider.ids["id"] = i
            slider.bind(value=self.on_value_change)

            sbox.add_widget(slider)
            sbox.add_widget(label)

            sboxes_ren += [(sbox, std)]
            # control.add_widget(sbox)

        sboxes_ren.sort(key=lambda x: x[1], reverse=True)
        for rsbox, _ in sboxes_ren:
            control.add_widget(rsbox)
        
        
        
        
        control_view = ScrollView(size=(Window.width, Window.height), size_hint=(1, 1))
        control_view.add_widget(control)
        
        
        self.root.add_widget(control_view)  
        self.root.add_widget(image)

        
    def build(self):
        file = open("mmms", "rb")
        data = file.read()
        self.mmms = np.frombuffer(data).reshape(50, 4)
        print(self.mmms, self.mmms.shape)
        file.close()
        self.input = np.zeros((1, 50), dtype=np.float32)
        
        for i, x in enumerate(self.mmms):
            self.input[0][i] = x[2]
        
            

        self.model = LoadModel(MODEL_NAME)

        self.view()
        self.update_texture()

        
        return self.root

TweakTok().run()