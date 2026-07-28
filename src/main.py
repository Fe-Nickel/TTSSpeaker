import customtkinter as ctk
import json
import tkinter as tk
from tkinter import filedialog,messagebox
import os
import sounddevice as sd
from datetime import datetime
import numpy as np
emo_vector_norm=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config=self.load_config()
        self.load_language(self.config.get("language"))
        self.EMOTIONS=[self.tr("happy"),self.tr("angry"),self.tr("sad"),self.tr("afraid"),self.tr("disgusted"),self.tr("depressed"),self.tr("surprised"),self.tr("neutral")]
        self.title(self.tr("window_title"))
        self.geometry("900x600")
        from indextts.infer_v2 import IndexTTS2
        self.tts=IndexTTS2(
            cfg_path=r"IndexTTS-2\config.yaml",
            model_dir=r"IndexTTS-2",
            use_fp16=True,
            use_cuda_kernel=False,)
        self.slider_vars={}
        self.slider_list=[]
        self.spk_dict_list=[]
        self.voice_name_list=[]
        for filename in os.listdir("spk_dicts"):
            if filename.endswith(".pt"):
                self.voice_name_list.append(os.path.splitext(filename)[0])
                self.spk_dict_list.append(self.tts.load_speaker_condition(f"spk_dicts/{filename}"))
        self.device_id=[]
        self.device_list=self.get_audio_devices()
        self.create_left_panel()
        self.create_right_panel()
        self.restore_config()
        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=1)
        self.protocol("WM_DELETE_WINDOW",self.on_close)
    def change_language(self,lang):
        self.load_language(lang)
        self.config["language"]=lang
        self.save_config()
        messagebox.showinfo(self.tr("tip"),self.tr("restart_required"))
    def load_language(self,lang_name):
        path=os.path.join("lang",f"{lang_name}.json")
        if not os.path.exists(path):
            path=os.path.join("lang","en_us.json")
        with open(path,"r",encoding="utf-8") as f:
            self.lang=json.load(f)
    def tr(self,key):
        return self.lang.get(key,key)
    def on_close(self):
        self.save_config()
        self.destroy()
    def restore_config(self):
        cfg=self.config
        if cfg["voice"] in self.voice_name_list:
            self.voice_var.set(cfg["voice"])
        else:
            self.voice_var.set("default")
        self.volume_var.set(cfg.get("volume",1.0))
        emotions=cfg.get("emotion",[0.0]*8)
        for slider,value in zip(self.slider_list[:8],emotions):
            slider.set(value)
    def load_config(self):
        if not os.path.exists("config.json"):
            return{
                "language":"en_us",
                "voice":"default",
                "volume":1.0,
                "emotion":[0.0]*8}
        try:
            with open("config.json","r",encoding="utf-8") as f:
                return json.load(f)
        except:
            return{
            "language":"en_us",
            "voice":"default",
            "volume":1.0,
            "emotion":[0.0]*8}
    def save_config(self):
        config={
        "language":self.language_var.get(),
        "voice":self.voice_var.get(),
        "volume":self.volume_var.get(),
        "emotion":[
            slider.get()
            for slider in self.slider_list[:8]]}
        with open("config.json","w",encoding="utf-8") as f:
            json.dump(config,f,ensure_ascii=False,indent=4)
    def create_left_panel(self):
        left_frame=ctk.CTkFrame(self)
        left_frame.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")
        left_frame.grid_columnconfigure(0,weight=1)
        left_frame.grid_rowconfigure(0,weight=1)
        left_frame.grid_rowconfigure(1,weight=0)
        self.chat_display=ctk.CTkScrollableFrame(left_frame)
        self.chat_display.grid(row=0,column=0,padx=5,pady=(5,0),sticky="nsew")
        input_frame=ctk.CTkFrame(left_frame)
        input_frame.grid(row=1,column=0,padx=5,pady=5,sticky="ew")
        input_frame.grid_columnconfigure(0,weight=1)
        self.msg_entry=ctk.CTkEntry(input_frame,placeholder_text=self.tr("input_placeholder"))
        self.msg_entry.grid(row=0,column=0,padx=(0, 5),sticky="ew")
        self.msg_entry.bind("<Return>",lambda e:self.send_message())
        self.send_btn=ctk.CTkButton(input_frame,text=self.tr("send"),width=80,command=self.send_message)
        self.send_btn.grid(row=0,column=1)
    def send_message(self):
        text=self.msg_entry.get().strip()
        if not text:
            return
        timestamp=datetime.now().strftime("%H:%M:%S")
        self.append_chat(f"[{timestamp}] {text}")
        self.msg_entry.delete(0,"end")
        if(self.monitor_var.get()):
            _index1=self.voice_name_list.index(self.voice_var.get())
            emo_vector=[
                slider.get()
                for slider in self.slider_list[:8]]
            audio_wav=self.tts.infer(
                spk_dict=self.spk_dict_list[_index1],
                text=text,
                emo_vector=emo_vector,
                use_random=False,
                emo_alpha=1.0,
                verbose=True)
            audio_np=audio_wav.detach().cpu().squeeze().numpy()
            _index2=self.device_id[self.device_list.index(self.device_var.get())]
            peak=max(abs(audio_np.max()),abs(audio_np.min()))
            audio_np/=peak
            audio_np*=self.volume_var.get()
            audio_np=np.clip(audio_np,-1,1)
            sd.play(audio_np,samplerate=22050,device=_index2)
    def get_audio_devices(self):
        devices=sd.query_devices()
        result=[]
        cnt=0
        for i,d in enumerate(devices):
            if(d["max_output_channels"]>0):
                result.append(f"{i}:{d['name']}")
                cnt=cnt+1
                self.device_id.append(i)
        return result
    def append_chat(self,message:str):
        bubble=ctk.CTkFrame(self.chat_display,fg_color="#2b5b84",corner_radius=12)
        bubble.pack(anchor="e",padx=10,pady=5)
        lbl=ctk.CTkLabel(bubble,
                        text=message,wraplength=300,
                        justify="left",
                        text_color="white",
                        font=("Arial",13))
        lbl.pack(padx=12,pady=8)
        self.after(10,self._scroll_to_bottom)
    def _scroll_to_bottom(self):
        canvas=getattr(self.chat_display,"_parent_canvas",None)
        if canvas:
            canvas.yview_moveto(1.0)
    def delete_voice(self):
        name=self.voice_var.get()
        if name in ("default",self.tr("import_voice")):
            messagebox.showwarning(self.tr("tip"),self.tr("tip_"))
            return
        result=messagebox.askyesno(self.tr("confirm_deletion"),self.tr("sure?"))
        if not result:
            return
        path=f"spk_dicts/{name}.pt"
        if os.path.exists(path):
            os.remove(path)
        index=self.voice_name_list.index(name)
        self.voice_name_list.pop(index)
        self.spk_dict_list.pop(index)
        self.voice_menu.configure(values=self.voice_name_list+[self.tr("import_voice")])
        self.voice_var.set("default")
        messagebox.showinfo(self.tr("done"),self.tr("voice_deleted"))
    def create_right_panel(self):
        right_frame=ctk.CTkFrame(self)
        right_frame.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")
        right_frame.grid_columnconfigure(0,weight=1)
        right_frame.grid_rowconfigure(0,weight=1)
        right_frame.grid_rowconfigure(1,weight=0)
        right_frame.grid_rowconfigure(2,weight=0)
        slider_frame=ctk.CTkScrollableFrame(right_frame,height=280)
        slider_frame.grid(row=0,column=0,padx=5,pady=5,sticky="nsew")
        slider_frame.grid_columnconfigure(0,weight=1)
        language_row=ctk.CTkFrame(slider_frame)
        language_row.pack(fill="x",pady=(0,10))
        ctk.CTkLabel(language_row,text=self.tr("language")).pack(side="left",padx=(0,5))
        self.language_var=ctk.StringVar(value=self.config.get("language","zh_cn"))
        self.language_menu=ctk.CTkOptionMenu(language_row,values=["zh_cn","en_us"],variable=self.language_var,command=self.change_language)
        self.language_menu.pack(side="left",fill="x",expand=True)
        voice_row=ctk.CTkFrame(slider_frame)
        voice_row.pack(fill="x",pady=(0,10))
        ctk.CTkLabel(voice_row,text=self.tr("voice")).pack(side="left",padx=(0,5))
        self.voice_var=ctk.StringVar(value="default")
        self.voice_menu=ctk.CTkOptionMenu(
            voice_row,
            values=self.voice_name_list+[self.tr("import_voice")],
            variable=self.voice_var,
            command=self.on_voice_selected)
        self.voice_menu.pack(side="left",fill="x",expand=True)
        delete_btn=ctk.CTkButton(
            voice_row,
            text=self.tr("delete_voice"),
            width=50,
            command=self.delete_voice)
        delete_btn.pack(side="left", padx=(10,5))
        extract_btn=ctk.CTkButton(
            voice_row,
            text=self.tr("extract_from"),
            width=50,
            command=self.open_extract_window)
        extract_btn.pack(side="left",padx=(10,0))
        self.slider_vars={}
        ctk.CTkLabel(slider_frame,text=self.tr("emotional_args"),anchor="w").pack(anchor="w",pady=(10, 2))
        for emotion in self.EMOTIONS:
            self.create_slider_row(slider_frame,emotion,0.0,1.0,0.0)
        # self.create_slider_row(slider_frame,"语速",0.5, 2.0,1.0)
        # self.create_slider_row(slider_frame,"音调",0.5,2.0,1.0)
        separator=ctk.CTkFrame(right_frame,height=2,fg_color="gray")
        separator.grid(row=1,column=0,padx=5,pady=5,sticky="ew")
        bottom_frame=ctk.CTkFrame(right_frame)
        bottom_frame.grid(row=2,column=0,padx=5,pady=5,sticky="ew")
        vol_frame=ctk.CTkFrame(bottom_frame)
        vol_frame.pack(fill="x",pady=(0,10))
        ctk.CTkLabel(vol_frame,text=self.tr("volume")).pack(side="left",padx=(0, 5))
        self.volume_var=ctk.DoubleVar(value=0.8)
        self.volume_slider=ctk.CTkSlider(
            vol_frame,from_=0,to=1,number_of_steps=100,
            variable=self.volume_var)
        self.volume_slider.pack(side="left",fill="x",expand=True)
        monitor_frame=ctk.CTkFrame(bottom_frame)
        monitor_frame.pack(fill="x",pady=(0,5))
        self.monitor_var=ctk.BooleanVar(value=False)
        self.monitor_switch=ctk.CTkSwitch(
            monitor_frame,text=self.tr("on_off"),variable=self.monitor_var,
            onvalue=True,offvalue=False)
        self.monitor_switch.pack(side="left",padx=(0,10))
        ctk.CTkLabel(monitor_frame,text=self.tr("Output Device/Virtual Audio Cable")).pack(side="left",padx=(0,5))
        self.device_var=ctk.StringVar(
            value=self.device_list[0] if self.device_list else self.tr("no_device"))
        self.device_menu=ctk.CTkOptionMenu(
            monitor_frame,
            values=self.device_list,
            variable=self.device_var)
        self.device_menu.pack(side="left",fill="x",expand=True)
    def on_voice_selected(self,choice:str):
        if(choice==self.tr("import_voice")):
            self.import_voice_file()
            return
    def import_voice_file(self):
        file_path=filedialog.askopenfilename(
            title=self.tr("select_voice"),
            filetypes=[("file","*.pt"),("all", "*.*")])
        if(not file_path):
            self.voice_var.set("default")
            return
        _name=os.path.splitext(os.path.basename(file_path))[0]
        current_values=list(self.voice_menu.cget("values"))
        if(self.tr("import_voice") in current_values):
            insert_idx=current_values.index(self.tr("import_voice"))
            new_values=current_values[:insert_idx]+[_name]+current_values[insert_idx:]
        else:
            new_values=current_values+[_name]
        self.voice_menu.configure(values=new_values)
        self.voice_var.set(_name)
        _temp_spk_dict=self.tts.load_speaker_condition(file_path)
        self.tts.save_speaker_condition(_temp_spk_dict,f"spk_dicts/{_name}.pt")
        self.spk_dict_list.append(_temp_spk_dict)
        self.voice_name_list.append(_name)
        messagebox.showinfo(self.tr("tip"),self.tr("import_success"))
    def open_extract_window(self):

        print(self)
        print(self.winfo_exists())
        print(tk._default_root)

        extract_win=ctk.CTkToplevel(self)
        extract_win.title(self.tr("extract_from"))
        extract_win.geometry("500x300")
        extract_win.grab_set()
        file_label=ctk.CTkLabel(extract_win,text=self.tr("no_file_selected"),fg_color="transparent")
        file_label.pack(pady=(20,10))
        def choose_audio():
            path=filedialog.askopenfilename(
                title=self.tr("select_audio"),
                filetypes=[("file","*.mp3 *.wav"), ("all","*.*")]
            )
            if path:
                file_label.configure(text=os.path.basename(path))
                self.extract_audio_path=path
        ctk.CTkButton(extract_win,text=self.tr("select_audio"),command=choose_audio).pack(pady=5)
        status_label=ctk.CTkLabel(extract_win,text="",text_color="gray")
        status_label.pack(pady=5)
        def start_extract():
            if not hasattr(self,'extract_audio_path'):
                status_label.configure(text=self.tr("please_select_audio"),text_color="red")
                return
            status_label.configure(text=self.tr("extracting"),text_color="orange")
            _temp_spk_dict=self.tts.extract_speaker_condition(self.extract_audio_path,verbose=True)
            _name=os.path.splitext(os.path.basename(self.extract_audio_path))[0]
            self.tts.save_speaker_condition(_temp_spk_dict,f"spk_dicts/{_name}.pt")
            self.spk_dict_list.append(_temp_spk_dict)
            self.voice_name_list.append(_name)
            new_values=self.voice_name_list+[self.tr("import_voice")]
            self.voice_menu.configure(values=new_values)
            self.voice_var.set(_name)
            extract_win.after(1000,lambda:status_label.configure(text=self.tr("done"),text_color="green"))
        ctk.CTkButton(extract_win,text=self.tr("start_extract"),command=start_extract).pack(pady=5)
    def create_slider_row(self,parent,label:str,from_val:float,to_val:float,default:float):
        row=ctk.CTkFrame(parent,fg_color="transparent")
        row.pack(fill="x",pady=2)
        lbl=ctk.CTkLabel(row,text=label,width=60,anchor="w")
        lbl.pack(side="left")
        var=ctk.DoubleVar(value=default)
        slider=ctk.CTkSlider(
            row,from_=from_val,to=to_val,number_of_steps=100,
            variable=var)
        slider.pack(side="left",fill="x",expand=True,padx=(5,0))
        self.slider_vars[label]=var
        self.slider_list.append(var)
app=App()
app.iconbitmap("icon.ico")
app.mainloop()