import json, requests
import time
import matplotlib.pyplot as plt
from PIL import Image
def tow():
    for i in range(324,1001):
        url ='https://api.vvhan.com/api/acgimg'
        params ={'type':'json'}
        res = requests.get(url, params=params).json()
        print(res)
        target_url=res['imgurl']
        img=requests.get(target_url).content
        print('will')
        with open(str(i)+'.jpg','wb') as image:
            image.write(img)
def er(i):
    # url='https://cdn.seovx.com/?mom=302'
    url='https://imgapi.xl0408.top/index.php'
    res=requests.get(url)
    print(i)
    with open('FindBeauty/img_er/'+str(i)+'.jpg','wb') as img:
        img.write(res.content)
    img = Image.open('FindBeauty/img_er/'+str(i)+'.jpg')
    plt.imshow(img)
def vive(i):

    for i in range(137,401):
        url='https://cdn.seovx.com/?mom=302'
        res=requests.get(url)
        print(i)
        with open('FindBeauty/img_vive/'+str(i)+'.jpg','wb') as img:
            img.write(res.content)
        img = Image.open('FindBeauty/img_vive/'+str(i)+'.jpg')
        plt.imshow(img)
        
def botian(i):
    url='https://api.btstu.cn/sjbz/api.php?lx=meizi'
    res=requests.get(url)
    with open('FindBeauty/img_vive/'+str(i)+'.jpg','wb') as img:
        img.write(res.content)
    img=Image.open('FindBeauty/img_vive/'+str(i)+'.jpg')
    plt.imshow(img)
plt.draw()
for i in range(508,551):
    plt.subplot(1,2,1)
    botian(i)
    plt.subplot(1,2,2)
    er(i)
    plt.pause(3)
    
