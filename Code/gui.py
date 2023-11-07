#pip install -r requirements.txt

from Path import *
from npytojpg import *

from tkinter import *
from PIL import ImageTk,Image

root=Tk()


def myclick():
    num=get_depth(path+"Sample_image\\0001.npy")
    number=e.get()
    myla2=Label(root,text=str(number)+' / '+str(num))
    myla2.grid(row=0,column=2)
    global slice
    slice=num

    get_image(path+"Sample_image\\0001.npy",34)
    #myla3.pack()

    #b()
#myLabel1.grid(row=1,column=1)

'''def b():
    myLabel2=Label(root,text=str(depth)+'')
    myLabel2.pack()'''
def display():
    img=ImageTk.PhotoImage(Image.open("C:\\Users\\bjaya\\Downloads\\application\\pancimage\\panc_img.jpg"))
    myla1=Label(image=img)
    myla1.grid(row=0,column=1,columnspan=4)

    mybutton=Button(root,text="Process",padx=20,pady=20,command=myclick(),fg="blue",bg="red")
    mybutton.grid(row=3,column=0)



e=Entry(root,width=30,borderwidth=7)
e.grid(row=0,column=0)
#e.pack()
#e.insert(0,"enter")
#e.delete(0,END)

#lambda:
mybutton=Button(root,text="Process",padx=20,pady=20,command=myclick(),fg="blue",bg="red")
mybutton.grid(row=3,column=0)



#button_quit=Button(root,text="Quit",command=root.quit)
#button_quit.grid(row=3,column=1)
#mylabelimg.pack()



root.mainloop()