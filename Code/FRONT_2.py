#pip install -r requirements.txt

from Path import *

from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import ImageTk,Image
from npytojpg import *
import tensorflow

root=Tk()

def store_img(directory_path):
    li=[]
    files = os.listdir(directory_path)
    for i in range(len(files)):
        img=ImageTk.PhotoImage(file=directory_path+"\\panc_img"+str(i)+".jpg")
        li.append(img)
    return li
org=store_img(path+'Resources\\input')
pred=store_img(path+'Resources\\output')
org_pred=store_img(path+'Resources\\input_output')

mylabel=Label(image=org[0])
mylabel.grid(row=2,column=2)

var=StringVar()
var2=StringVar()

checkbox=Checkbutton(root,text="MRI",variable=var,onvalue="on",offvalue="off").grid(row=6,column=1)
checkbox1=Checkbutton(root,text="Predicted",variable=var2,onvalue="on",offvalue="off").grid(row=6,column=2)

def prev(img_number):
    global org
    global org_pred
    global pred
    global mylabel
    global page_num
    global button_prev
    global button_next

    if(img_number>0):
        mylabel.grid_forget()
        if(var.get()=="on" and var2.get()=="on"):
            mylabel=Label(image=org_pred[(img_number-1)])
        elif(var2.get()=="on"):
            mylabel=Label(image=pred[(img_number-1)])
        else:
            mylabel=Label(image=org[(img_number-1)])
        mylabel.grid(row=2,column=2)
        button_prev=Button(root,text="Prev",command=lambda:prev(img_number-1)).grid(row=4,column=0)
        button_next=Button(root,text="Next",command=lambda:next(img_number-1)).grid(row=4,column=2)
        page_num=Label(root,text="Page number: "+str(img_number)+"/"+str(len(org)))
        page_num.grid(row=1,column=2)

def next(img_number):
    global org
    global org_pred
    global pred
    global mylabel
    global page_num
    global button_prev
    global button_next

    if(img_number<len(org)-1):
        mylabel.grid_forget()
        mylabel.grid_forget()
        if(var.get()=="on" and var2.get()=="on"):
            mylabel=Label(image=org_pred[(img_number+1)])
        elif(var2.get()=="on"):
            mylabel=Label(image=pred[(img_number+1)])
        else:
            mylabel=Label(image=org[(img_number+1)])
        mylabel.grid(row=2,column=2)
        button_prev=Button(root,text="Prev",command=lambda:prev(img_number+1)).grid(row=4,column=0)
        button_next=Button(root,text="Next",command=lambda:next(img_number+1)).grid(row=4,column=2)
        page_num=Label(root,text="Page number: "+str(img_number+2)+"/"+str(len(org)))
        page_num.grid(row=1,column=2)


button_prev=Button(root,text="Prev",command=lambda:prev(0)).grid(row=4,column=0)
button_next=Button(root,text="Next",command=lambda:next(0)).grid(row=4,column=2)


page_num=Label(root,text="Page number: 1/"+str(len(org)))
page_num.grid(row=1,column=2)

title=Label(root,text="Pancreas Segmentation")
title.grid(row=0,column=2)

button_quit=Button(root,text="Quit",command=root.quit).grid(row=4,column=1)


root.mainloop()