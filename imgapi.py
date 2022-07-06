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
def vive():
    plt.draw()
    for i in range(137,401):
        url='https://cdn.seovx.com/?mom=302'
        res=requests.get(url)
        print(i)
        with open('FindBeauty/img_vive/'+str(i)+'.jpg','wb') as img:
            img.write(res.content)
        img = Image.open('FindBeauty/img_vive/'+str(i)+'.jpg')
        plt.imshow(img)
        plt.pause(3)
        
        
vive()