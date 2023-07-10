import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from collections import Counter


def encode(image,block_size,codes_size):
    size=[block_size,block_size]
    img_temp=[]
    r1,c1=image.shape
    for j in range(c1):
        temp=[]
        for i in range(r1):
            temp.append(image[i][j])
        img_temp.append(temp)
    r,c=image.shape
    r2=r
    c2=c
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
    for k in range(int(x)):
        for t in range(int(y)):
            temp=[]
            for i in range(size[0]):
                for j in range(size[1]):
                    temp.append(img[x1+i][y1+j])
            new_img.append(temp)
            y1+=size[1]
        y1=0
        x1+=size[0]

    output=[]
    l=0
    max_value=0
    number_of_codes=0
    for x in range(len(new_img)):
        dic={f"{i}":i  for i in range(256) }
        p=""
        cc=""
        p+=str(new_img[x][0])
        last=256
        array=[]
        for i in range(len(new_img[x])):
            if(i != len(new_img[x])-1):
                cc+=str(new_img[x][i+1])
            
            if str(p+" "+cc) in dic:
                p=p+" "+cc
            else:
                # print(p,dic[p],"    ",p+" "+c,last)
                array.append(dic[p])
                max_value=max(max_value,dic[p])
                number_of_codes+=1
                if(last<2**codes_size):
                    dic[p+" "+cc]=last
                    last+=1
                p=cc
            cc=""
        output.append(array)
    
    entropy_arr=[]
    file1 = open("lzw.txt","w")
    file1.write(str(r2)+" "+str(c2)+" "+str(block_size)+"\n")
    compressed_img=[]
    for k in range(len(output)):
        for i in range(len(output[k])):
            compressed_img.append(output[k][i])
            entropy_arr.append(output[k][i])
            file1.write(str(output[k][i]))
            file1.write(" ")
        compressed_img.append("$")
        file1.write("\n")
    file1.close()

    counts= Counter(entropy_arr)
    total_symbols = sum(counts.values())
    arr= np.array(list(counts.values()))
    probabilities = arr / total_symbols
    entropy = -np.sum(probabilities * np.log2(probabilities))


    return compressed_img,max_value,number_of_codes,(r*c)/number_of_codes,(number_of_codes/len(new_img))*code_size,entropy


def decode(file_name):
    file1 = open(file_name,"r")
    contents=file1.readline()
    array=[]
    for line in file1:
        if(line[len(line)-1]=="\n"):
            array.append(line[:-1])
        else:
            array.append(line)
    contents=contents.split()
    r=int(contents[0])
    c=int(contents[1])
    r1=r
    c1=c
    size=int(contents[2])
    file1.close
    arr=[]
    for x in range(len(array)):
        sp_arr=array[x].split()
        arr.append(sp_arr)
    new_img=[]
    for x in range(len(arr)):

        dic={}
        for i in range(256):
            dic[str(i)]=str(i)

        temp=arr[x][0]
        temp_2=dic[temp]
        temp_3=(temp.split())[0]
        result=""
        result+=temp_2+" "
        last=256
        for i in range(len(arr[x])-1):
            temp_4=arr[x][i+1]
            if str(temp_4) not in dic:
                temp_2=dic[temp]
                temp_2=temp_2+" "+temp_3
            else:
                temp_2=dic[str(temp_4)]
            result+=temp_2+" "
            temp_3=""
            temp_3+=(temp_2.split())[0]
            dic[str(last)]=dic[temp]+" "+temp_3
            last+=1
            temp=str(temp_4)
            
        result=result.split()
        
        array_temp=[]
        j=0
        while (j<size*size):
            arr1=[]
            for i in range(size):
                arr1.append(result[i+j])
            array_temp.append(arr1)
            j+=size
        array_temp=np.uint8(array_temp)
        new_img.append(array_temp)

    x=int(r/size)
    y=int(c/size)
    if(r%size!=0):
        x+=1
    if c%size!=0:
        y+=1
    if(size-r%size!=size):
        r+=size-r%size
    if(size-c%size!=size):
        c+=size-c%size

    final_img=np.zeros((r,c),dtype="uint8")
    
    ind=0
    for j in range(x):
        for i in range(y):
            for x1 in range(size):
                for y1 in range(size):
                    final_img[x1+j*size][y1+i*size]=new_img[ind][x1][y1]
            ind+=1
    
    output_image=np.zeros((r1,c1), dtype=np.uint8)

    for i in range(r1):
        for j in range(c1):
            output_image[i][j]=final_img[i][j]

    return output_image


if __name__ == "__main__":


    image_no=int(input("Enter the image number which you want to open(1-24) :  "))
    file_name=f"Images/{image_no}.tif"
    img=cv.imread(file_name)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    block_size=int(input("Enter the Block Size (8,16,32,etc) .Enter -1 for whole image :  "))
    code_size=int(input("Enter the Code Size (9,10,12,etc) .Enter -1 for default value 16 :  "))
    if(code_size==-1):
        code_size=16
    if(block_size==-1):
        block_size=min(img.shape[0],img.shape[1])


    comp_img,max_value,number_of_codes,CR,avg_len,entropy=encode(img,block_size,code_size)

    print("Max value of any code used: ",max_value)
    print("Total number of codes : ",number_of_codes)
    print("Compression Ratio achieved : ",CR)
    print("Average length of encoded pixels (average number of bits for each block) : ",avg_len)
    print("Entropy : ",entropy)


    img1=(decode("lzw.txt"))

    RMSE=round(math.sqrt(np.mean((img1-img)**2)),5)
    print("RMSE : ",RMSE)
    plt.figure(figsize=[18,5])
    plt.subplot(1,2,1);plt.imshow(img,cmap="gray");plt.title("Original Image");
    plt.subplot(1,2,2);plt.imshow(img1,cmap="gray");plt.title("Image after decoding");
    plt.savefig("output/lzw.png")
    plt.show()