#pip install -r requirements.txt


from Path import *

from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import ImageTk,Image
from npytojpg import *
import tensorflow
from multiprocessing import Process
import subprocess

root1=Tk()


img=ImageTk.PhotoImage(Image.open(path+'Resources\\display_img.jpg'))

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

def checkpath():
    filetypes = (
        ('text files', '*.npy'),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(title='Open a file',filetypes=filetypes)
    global open_path
    open_path=str(filename)
    

    try:
        global depth
        depth= get_depth(open_path)
        
        delete_files_in_directory(path+'Resources\\input')

        delete_files_in_directory(path+'Resources\\output')
        delete_files_in_directory(path+'Resources\\input_output')
        
        img = np.load(open_path)
        img = img.transpose()
        get_image(img,path+'Resources\\input\\panc_img')
        print('now predict')
        img1=predict(img)
        print(img1.shape)
        get_image(img1,path+'Resources\\output\\panc_img')

        get_image2(img,img1,path+'Resources\\input_output\\panc_img')

        subprocess.run(['python', 'FRONT_2.py'])
    except:
        return
    

def front2():
    subprocess.run(['python', 'FRONT_2.py'])

title=Label(root1,text="Pancreas Segmentation")
title.grid(row=0,column=2)

myla3=Label(image=img)
myla3.grid(row=0,column=3)
button_display=Button(root1,text="Display",command=front2).grid(row=3,column=3)
button_quit=Button(root1,text="Quit",command=root1.quit).grid(row=3,column=1)


open_button = ttk.Button(
    root1,
    text='Open a File',
    command=checkpath
)

open_button.grid(row=3,column=2)


root1.mainloop()