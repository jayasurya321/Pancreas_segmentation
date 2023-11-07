#pip install -r requirements.txt


from tkinter import *
from PIL import ImageTk,Image
from npytojpg import *
from Path import *

#root=Tk()

def display():
    global depth
    t=[]
    for i in range(depth):
        img=ImageTk.PhotoImage(file=path+'Resources\\panc_img'+str(i)+'.jpg')
        t.append(img)

    myla2=Label(root,image=t[3])
    myla2.grid(row=0,column=2)

'''
def myclick():
    ans=str(e.get())
    if ans=='':
        ans='10'
    ans=int(ans)
    myla2=Label(root,text=' / '+str(type(ans))+str(ans))
    myla2.grid(row=0,column=2)
    get_image("C:\\Users\\bjaya\\Downloads\\application\\0001np\\0001.npy",ans)
    #img=ImageTk.PhotoImage(Image.open('C:\\Users\\bjaya\\Downloads\\application\\pancimage\\panc_img.jpg'))
    #myla3=Label(image=img)
    #myla3.grid(row=0,column=3)
    #myla3.pack()
    display()


def display():
    myla1=Label(image=img)
    myla1.grid(row=0,column=1,columnspan=4)

    #mybutton=Button(root,text="Process",padx=20,pady=20,command=myclick(),fg="blue",bg="red")
    #mybutton.grid(row=3,column=0)

mybutton=Button(root,text="Process",padx=20,pady=20,command=myclick(),fg="blue",bg="red")
mybutton.grid(row=3,column=0)
'''


#root.mainloop()