import cv2
import numpy as np
import numpy.linalg as la
from tkinter import StringVar, Tk, Button, Label, Frame, messagebox, filedialog
from tkinter.constants import CENTER, RIGHT, LEFT
from tkinter.ttk import * 

image_path = ''
U, sigma, V_T = None, None, None
max_rank = 0
#==============================================================================================================================================
def eig_calculation(A):
    '''
        Calculate the eigenvalues and eigenvectors of matrix A^T.A 
        Arguments:
            A: numpy array (the image)

        Returns:
            eigenvalues: numpy array
            eigenvectors: numpy array
    '''
    AT_A = np.dot(A.T, A)
    eigenvalues, eigenvectors = la.eigh(AT_A)
    eigenvalues = np.maximum(eigenvalues, 0.)

    sorted_index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:, sorted_index]

    return eigenvalues, eigenvectors

def calculation(A):
    '''
        Using SVD to calculate U, sigma and V^T matrices of matrix A
        Arguments:
            A: numpy array, the image

        Returns:
            U: numpy array
            sigma: numpy array
            V_T: numpy array
    '''
    eigenvalues, eigenvectors = eig_calculation(A)
    
    sigma = np.zeros([A.shape[0], A.shape[1]])
    for i in range(min(A.shape[0], A.shape[1])):
        sigma[i][i] = max(eigenvalues[i], 0.)
    sigma = np.maximum(np.sqrt(sigma), 0)

    V = eigenvectors
    V_T = V.T
    
    U = np.zeros([A.shape[0],A.shape[0]])
    for i in range(A.shape[0]):
        U[:, i] = 1/sigma[i][i] * np.dot(A, V[:, i])

    return U, sigma, V_T 

def create_A_approx(U, sigma, V_T, rank):
    '''
        Calculate the matrix A approximately from matrices U, sigma, V_T and rank using SVD
        Arguments:
            U: numpy array
            sigma: numpy array
            V_T: numpy array
            rank: int, the rank of the approximate matrix, 
                the greater the rank is the more accuracy the approximate image is

        Returns:
            result: numpy array, the approximately image
            error: double, the error of the approximate image
    '''
    result = np.zeros(sigma.shape)
    for i in range(rank):
        u = np.reshape(np.array(U.T)[i], (sigma.shape[0], 1))
        v = np.reshape(np.array(V_T)[i], (1, sigma.shape[1]))
        result += sigma[i][i] * np.dot(u, v)
    error = sigma[rank-1][rank-1] ** 2
    return result, error
    
#==============================================================================================================================================
def makeCenter(root):
    '''
        Place the window of the app in the center of the screen
        Arguments:
            root: a tkinter instance
        
        Returns:
            None
    '''
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth()//2) - (width//2)
    y = (root.winfo_screenheight()//2) - (height//2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def get_input_file():
    '''
        Get image path from user
        Arguments:
            None
        Returns:
            None
    '''
    global image_path
    image_path = filedialog.askopenfilename(initialdir="D:", title = "Select a file", filetypes = (("all files", "*.*"), ("image files", "*.jpg"), ("image files", "*.png"), ("image files", "*.jpeg")))
    if len(image_path)!=0: 
        input_label.config(text=image_path, foreground='green')
    else:
        input_label.config(text="Please choose input file", foreground='red')

def process():
    '''
        Process the image to calculate U, sigma, V_T and max_rank
        Arguments:
            None

        Returns:
            None
    '''
    global U, sigma, V_T, max_rank
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray/255.
        gray = np.array(gray)
        U, sigma, V_T = calculation(gray)
        max_rank = la.matrix_rank(sigma)
        process_label.config(text=str(max_rank))
    except Exception as e:
        print(e)
        messagebox.showerror('Error', 'This is not an image, please try again !!!')

def show_image():
    '''
        Get rank from user to calculate and show approximate image
        Arguments:
            None
            
        Returns:
            None
    '''
    global U, sigma, V_T
    try:
        rank = rank_value.get()
        rank = int(rank)
        A_approx, error = create_A_approx(U, sigma, V_T, rank)
        error_label.config(text='Error: ' + str(error))
        cv2.imshow('Approximate image', A_approx)
        cv2.waitKey(0)
    except:
        messagebox.showerror('Error', 'Rank must be a positive integer <= '+ str(max_rank) +' !!!')

#==============================================================================================================================================
root = Tk()
root.title('SVD')
root.geometry("600x250")
makeCenter(root)
root.resizable(width=False, height=False)
label = Label(root, text='Image reconstruction', font=(20)).pack(pady=5)

input_frame = Frame(root)
input_button = Button(input_frame, width=20, text = 'Choose image', command=get_input_file)
input_button.pack(side=LEFT)
input_label = Label(input_frame, width=60)
input_label.pack()
input_frame.pack(pady=10)

process_frame = Frame(root)
process_button = Button(process_frame, width=20, text = 'Process image', command=process)
process_button.pack(side=LEFT)
process_label = Label(process_frame, width=60, foreground='green')
process_label.pack()
process_frame.pack(pady=10)

rank_frame = Frame(root)
rank_value = StringVar()
rank_label = Label(rank_frame, width=20, text='Rank', anchor=CENTER)
rank_label.pack(side=LEFT)
rank_entry = Entry(rank_frame, width=40, textvariable=rank_value)
rank_button = Button(rank_frame, width=20, text='OK', command=show_image)
rank_button.pack(side=RIGHT)
rank_entry.pack()
rank_frame.pack(pady=10)

error_label = Label(root)
error_label.pack()

root.mainloop()