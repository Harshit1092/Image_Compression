import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os

def getarray(image,size,no_of_coeff,matrix):
    r,c=image.shape
    img_temp=[]
    for j in range(c):
        temp=[]
        for i in range(r):
            temp.append(image[i][j])
        img_temp.append(temp)

    x=int(r/size[0])
    y=int(c/size[1])
    if(r%size[0]!=0):
        x+=1
    if c%size[1]!=0:
        y+=1

    if(size[1]-c%size[1]!=size[1]):
        for i in range(size[1]-c%size[1]):
            temp=[0 for i in range(r)]
            img_temp.append(temp)
        c+=size[1]-c%size[1]

    if(size[0]-r%size[0]!=size[0]):
        for j in range(c): 
            for i in range(size[0]-r%size[0]):
                img_temp[j].append(0)
        r+=size[1]-r%size[0]

    img=[]
    for j in range(r):
        temp=[]
        for i in range(c):
            temp.append(img_temp[i][j])
        img.append(temp)

    new_img=[]
    x1=0
    y1=0

    for k in range(int(y)):
        for t in range(int(x)):
            temp=np.zeros((size[0],size[1]),dtype=np.int8)
            # print(x1,y1,x,y)
            for i in range(size[0]):
                for j in range(size[1]):
                    if(x1+i>r-1 or y1+j>c-1):
                        print(x1,y1,x,y,i,j)
                        continue
                    else:
                        temp[i,j]=img[x1+i][y1+j]-128
            new_img.append(temp)
            x1+=size[0]
        x1=0
        y1+=size[1]

    img_dct_arr=[]

    for k in range(len(new_img)):
        imf = np.float32(new_img[k])
        img_dct = cv.dct(imf, cv.DCT_INVERSE)
        img_dct=img_dct/matrix
        img_dct = np.rint(img_dct)
        img_dct_arr.append(img_dct)

    new_img_arr=[]
    for k in range(len(img_dct_arr)):
        arr=[[] for i in range(size[0]+size[1]-1)]

        for i in range(size[0]):
            for j in range(size[1]):
                sum=i+j
                if(sum%2 ==0):
                    arr[sum].insert(0,img_dct_arr[k][i][j])
                else:
                    arr[sum].append(img_dct_arr[k][i][j])
        temp_arr=[]
        for i in arr:
            for j in i:
                temp_arr.append(j)
        new_img_arr.append(temp_arr)
    
    if(no_of_coeff!=-1):
        for k in range(len(new_img_arr)):
            new_img_arr[k]=new_img_arr[k][:no_of_coeff]
    else:
        for k in range(len(new_img_arr)):
            while(len(new_img_arr[k])>1 and new_img_arr[k][len(new_img_arr[k])-1]==0):
                new_img_arr[k].pop()

    compressed_img=[]
    file1 = open("MyFile.txt","a")
    for k in range(len(new_img_arr)):
        for i in range(len(new_img_arr[k])):
            compressed_img.append(new_img_arr[k][i])
            file1.write(str(new_img_arr[k][i]))
            file1.write(" ")
        compressed_img.append("$")
        file1.write("\n")
    file1.write("\n&&&&&&&&&&\n\n")
    file1.close()
    return compressed_img

def encoding(name,no_of_coeff,size,matrix,iscolor):
    image=cv.imread(name)
    if(iscolor==True):
        image = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
        y,u,v = cv.split(image)
        file1 = open("MyFile.txt","w")
        file1.close
        array_y=getarray(y,size,no_of_coeff,matrix)
        array_u=getarray(u,size,no_of_coeff,matrix)
        array_v=getarray(v,size,no_of_coeff,matrix)
        array=array_y+["&&"]+array_u+["&&"]+array_v
    else:
        file1 = open("MyFile.txt","w")
        file1.close
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        array=getarray(image,size,no_of_coeff,matrix)
    return array

def get_image(name,array,size,quant,r,c,x,y):
    new_decoded_img=[]
    for p in range(len(array)):
        temp2=[]
        sizee=1
        # print(arr1)

        j=0
        while(sizee<=max(size[0],size[1])):
            temp1=[]
            t=j
            while(j<t+sizee):
                temp1.append(array[p][j])
                j+=1
            temp2.append(temp1)
            sizee+=1
        sizee-=1
        for i in range(max(size[0],size[1])-min(size[0],size[1])):
            temp1=[]
            t=j
            while(j<t+sizee):
                temp1.append(array[p][j])
                j+=1
            temp2.append(temp1)
        sizee-=1
        while(sizee>0):
            temp1=[]
            t=j
            while(j<t+sizee):
                temp1.append(array[p][j])
                j+=1
            temp2.append(temp1)
            sizee-=1
        matrix=[["." for i in range(size[1])] for i in range(size[0])]

        for k in range(int((len(temp2)+1)/2)):
            if(k%2!=0):
                i=0
                j=len(temp2[k])-1
                while(j>=0):
                    matrix[i][j]=temp2[k][i]
                    j-=1
                    i+=1
            else:
                j=0
                i=len(temp2[k])-1
                while(i>=0):
                    matrix[i][j]=temp2[k][j]
                    i-=1
                    j+=1

        for k in range(int((len(temp2))/2)):
            if((len(temp2)-k-1)%2!=0):
                i=0
                j=len(temp2[len(temp2)-k-1])-1
                while(j>=0):
                    matrix[size[0]-i-1][size[1]-j-1]=temp2[len(temp2)-k-1][i]
                    j-=1
                    i+=1
            else:
                j=0
                i=len(temp2[len(temp2)-k-1])-1
                while(i>=0):
                    matrix[size[0]-i-1][size[1]-j-1]=temp2[len(temp2)-k-1][j]
                    i-=1
                    j+=1
        quant = np.float32(quant)
        matrix*=quant
        matrix=cv.idct(matrix)
        matrix+=128
        matrix=np.rint(matrix)
        matrix=np.uint8(matrix)
        new_decoded_img.append(matrix)
    final_decoded_img=np.zeros((r,c), dtype=np.uint8)
    x1=0
    y1=0
    for k in range(len(new_decoded_img)):
        if(x1==x):
            x1=0
            y1+=1
        for i in range(size[0]):
            for j in range(size[1]):
                final_decoded_img[i+x1*size[0]][j+y1*size[1]]=new_decoded_img[k][int(i)][int(j)]
        
        x1+=1

    imggg=cv.imread(name)
    r,c=imggg.shape[0],imggg.shape[1]

    output_image=np.zeros((r,c), dtype=np.uint8)

    for i in range(r):
        for j in range(c):
            output_image[i][j]=final_decoded_img[i][j]

    return output_image

def decoding(name,size,quant,iscolor,compressed_img):
    if(iscolor==False):
        ind=0
        array=[]
        while(ind<len(compressed_img)):
            arr1=[]
            i=0
            while(compressed_img[i+ind] != "$"):
                arr1.append(compressed_img[i+ind])
                if(compressed_img[i+ind] == "$"):
                    break
                i+=1
            ind+=i+1
            array.append(arr1)
        for i in range(len(array)):
            while(len(array[i])<size[0]*size[1]):
                array[i].append(0.0)


        image=cv.imread(name)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        r,c=image.shape
        x=int(r/size[0])
        y=int(c/size[1])
        if(r%size[0]!=0):
            x+=1
        if c%size[1]!=0:
            y+=1
        r+=size[0]-r%size[0]
        c+=size[1]-c%size[1]

        final_img=get_image(name,array,size,quant,r,c,x,y)
    else:
        ind=0
        array_y=[]
        while(ind<len(compressed_img)):
            arr1=[]
            i=0
            cnd1=1
            while(compressed_img[i+ind] != "$"):
                if(compressed_img[i+ind]=="&&"):
                    cnd1=0
                    break
                arr1.append(compressed_img[i+ind])
                if(compressed_img[i+ind] == "$"):
                    break
                i+=1
            ind+=i+1
            if(cnd1==0):
                break
            array_y.append(arr1)

        for i in range(len(array_y)):
            while(len(array_y[i])<size[0]*size[1]):
                array_y[i].append(0.0)

        array_u=[]
        while(ind<len(compressed_img)):
            arr1=[]
            i=0
            cnd1=1
            while(compressed_img[i+ind] != "$"):
                if(compressed_img[i+ind]=="&&"):
                    cnd1=0
                    break
                arr1.append(compressed_img[i+ind])
                if(compressed_img[i+ind] == "$"):
                    break
                i+=1
            ind+=i+1
            if(cnd1==0):
                break
            array_u.append(arr1)

        for i in range(len(array_u)):
            while(len(array_u[i])<size[0]*size[1]):
                array_u[i].append(0.0)

        array_v=[]
        while(ind<len(compressed_img)):
            arr1=[]
            i=0
            cnd1=1
            while(compressed_img[i+ind] != "$"):
                if(compressed_img[i+ind]=="&&"):
                    cnd1=0
                    break
                arr1.append(compressed_img[i+ind])
                if(compressed_img[i+ind] == "$"):
                    break
                i+=1
            ind+=i+1
            if(cnd1==0):
                break
            array_v.append(arr1)

        for i in range(len(array_v)):
            while(len(array_v[i])<size[0]*size[1]):
                array_v[i].append(0.0)

        

        image=cv.imread(name)
        r,c,ch=image.shape
        x=int(r/size[0])
        y=int(c/size[1])
        if(r%size[0]!=0):
            x+=1
        if c%size[1]!=0:
            y+=1
        r+=size[0]-r%size[0]
        c+=size[1]-c%size[1]

        final_img1=get_image(name,array_y,size,quant,r,c,x,y)
        final_img2=get_image(name,array_u,size,quant,r,c,x,y)
        final_img3=get_image(name,array_v,size,quant,r,c,x,y)
        final_img=cv.merge((final_img1,final_img2,final_img3))
        final_img=np.uint8(final_img)
        final_img=cv.cvtColor(final_img, cv.COLOR_YCR_CB2BGR)

    return final_img

def print_stats(array,img,img1,color):
    print("PRINTING STATS:- ")
    if(color==True):
        n=len(array)
        Cr=(img1.shape[0]*img1.shape[1]*3)/n
        R=1-(1/Cr)
    else:
        n=len(array)
        Cr=(img1.shape[0]*img1.shape[1])/n
        R=1-(1/Cr)

    if(color==False):
        img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

    RMSE=round(math.sqrt(np.mean((img1-img)**2)),5)
    print("RMSE: ",RMSE)
    print("Compresion ratio: ",Cr)
    print("Redundancy : ",1-(1/Cr))
    PSNR=20*(math.log((255/RMSE),10))
    print("PSNR: ",PSNR)

def print_graph():
    directory = 'test/'

    average_psnr=[]
    average_cr=[]

    num_coeff=[1,3,6,10,15,28]

    for i in range(len(num_coeff)):
        no_of_coeff=num_coeff[i]
        psnr_arr=[]
        cr_arr=[]
        sum_psnr=0
        sum_cr=0
        total_terms=0
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')or filename.endswith('.tif'):
                image=cv.imread("test/"+str(filename))
                size=[8,8]
                iscolor=True
                quant=[[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
                encode=encoding("test/"+str(filename),no_of_coeff,size,quant,iscolor)
                decode=decoding("test/"+str(filename),size,quant,iscolor,encode)
                RMSE=round(math.sqrt(np.mean((image-decode)**2)),5)
                PSNR=20*(math.log((255/RMSE),10))
                psnr_arr.append(PSNR)
                sum_psnr+=PSNR
                total_terms+=1
                if(color==True):
                    n=len(encode)
                    Cr=(image.shape[0]*image.shape[1]*3)/n
                else:
                    n=len(encode)
                    Cr=(image.shape[0]*image.shape[1])/n
                cr_arr.append(Cr)
                sum_cr+=Cr
                print("Image Name: ",filename,"   ,  No. of coefficients: ",no_of_coeff,"   ,  PSNR: ",PSNR,"   ,  Compression Ratio: ",Cr)
        average_psnr.append(sum_psnr/total_terms)
        average_cr.append(sum_cr/total_terms)
    
    func, axis = plt.subplots()
    axis.plot(average_cr, average_psnr, marker='o')
    axis.set_title('PSNR vs CR graph')
    axis.set_xlabel('CR')
    axis.set_ylabel('PSNR')
    func.savefig('graphs/PSNRvsCR_graph.png')


def print_graph_analysis():
    filename="1.png"
    num_coeff=[1,3,6,10,15,28]
    image=cv.imread("test/"+str(filename))
    size=[8,8]
    iscolor=True
    quant=[[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
    psnr_arr=[]
    cr_arr=[]
    rmse_arr=[]
    psnr_max=0
    max_psnr=[]
    for i in range(len(num_coeff)):
        no_of_coeff=num_coeff[i]
        encode=encoding("test/"+str(filename),no_of_coeff,size,quant,iscolor)
        decode=decoding("test/"+str(filename),size,quant,iscolor,encode)
        RMSE=round(math.sqrt(np.mean((image-decode)**2)),5)
        PSNR=20*(math.log((255/RMSE),10))
        psnr_arr.append(PSNR)
        rmse_arr.append(RMSE)
        psnr_max=max(psnr_max,PSNR)
        if(color==True):
            n=len(encode)
            Cr=(image.shape[0]*image.shape[1]*3)/n
        else:
            n=len(encode)
            Cr=(image.shape[0]*image.shape[1])/n
        cr_arr.append(Cr)
        print("Image Name: ",filename,"   ,  No. of coefficients: ",no_of_coeff,"   ,  PSNR: ",PSNR,"   ,  Compression Ratio: ",Cr)
    for i in range(len(psnr_arr)):
        max_psnr.append(psnr_arr[i]/psnr_max)
    plt.plot(num_coeff, psnr_arr, marker='o',label="PSNR")
    plt.plot(num_coeff, cr_arr, marker='o',label="Compression ratio")
    plt.plot(num_coeff, rmse_arr, marker='o',label="RMSE")
    plt.plot(num_coeff, max_psnr, marker='o',label="Normalised PSNR")
    plt.title('PSNR,CR,RMSE,normalised PSNR vs no of coeff graph')
    plt.xlabel('Number of coefficients')
    plt.ylabel('')
    plt.legend()
    plt.savefig('graphs/PSNRvsCR_graph.png')



if __name__ == "__main__":

    image_no=int(input("Enter the image number which you want to open(1-24) :  "))
    file_name=f"test1/{image_no}.png"
    img1=cv.imread(file_name)

    inp_color=int(input("Color image or not?(1 for color/0 for grayscale) :  "))
    if(inp_color==1):
        color=True
    else:
        color=False

    block_size=int(input("Enter the Block Size (8,16,32,etc) .Enter -1 for whole image :  "))
    num_coeff=int(input("Enter the Number of coefficients (1,2,3,etc) .Enter -1 for all coefficients :  "))
    if(block_size==-1):
        block_size=min(img1.shape[0],img1.shape[1])
    block_size=[block_size,block_size]

    quant=[[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
    quant=np.array(quant)
    quant= cv.resize(quant,(block_size[0],block_size[1]),interpolation=cv.INTER_NEAREST)

    # inp_graph=int(input("Do you want to print the graph?(1 for YES/0 for NO) :  "))
    # if(inp_graph==1):
    #     print_graph()

    array=encoding(name=file_name,no_of_coeff=num_coeff,size=block_size,matrix=quant,iscolor=color)

    img=decoding(name=file_name,size=block_size,quant=quant,iscolor=color,compressed_img=array)

    print_stats(array,img,img1,color)

   
    if(color==True):
        fig=plt.figure(figsize=[18,5])
        plt.subplot(1,2,1);plt.imshow(img1[...,::-1]);plt.title("Original Image");
        plt.subplot(1,2,2);plt.imshow(img[...,::-1]);plt.title("Image Recovered from Decoding");
    else:
        img_1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        fig=plt.figure(figsize=[18,5])
        plt.subplot(1,2,1);plt.imshow(img_1,cmap="gray");plt.title("Original Image");
        plt.subplot(1,2,2);plt.imshow(img,cmap="gray");plt.title("Image Recovered from Decoding");

    plt.show()

    fig.savefig(f'output/output.png')
    # print(array)