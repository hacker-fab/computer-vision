import cv2
import numpy as np

imgs=["etch.jpg","etch1.jpg","etch2.jpg","etch3.jpg"]
for img in imgs:
    print("img %s" %img)
    org = cv2.imread(img)

    # Resize
    org=cv2.resize(org, None, fx=0.25, fy=0.25)
    oh=org.shape[0]

    # Convert the image to grayscale
    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", gray)
    cv2.waitKey(0)

    # Canny edge detection
    canny=cv2.Canny(gray,20,80)
    cv2.imshow("img", canny)
    cv2.waitKey(0)

    # Apply dilate to bold edge outline
    kernel_size = (4, 4)
    im1 = cv2.dilate(canny, np.ones(kernel_size, dtype=np.uint8), iterations=1)
    cv2.imshow("img", im1)
    cv2.waitKey(0)

    template = cv2.imread('template.jpg',0)
    th,tw=template.shape
    res = cv2.matchTemplate(im1,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.34
    loc = np.where(res >= threshold)
    cross_center=[]#array of [x,y,n], (x,y) is average of n near(distance((x,y),p)<th//5) points
    for pt in zip(*loc[::-1]):
#       cv2.rectangle(org, pt, (pt[0] + tw, pt[1] + th), (255,0,0), 1)
        x=pt[0]+tw/2
        y=pt[1]+th/2
        if y<oh/2: continue #Ignore matching in the upper half of the screen
        found=False
        for p in cross_center:
            if (p[0]-x)*(p[0]-x)+(p[1]-y)*(p[1]-y)<(th//5)*(th//5):#distance((x,y),p)<th//5
                found=True
                n=p[2]
                p[0]=(p[0]*n+x)/(n+1)
                p[1]=(p[1]*n+y)/(n+1)
                p[2]=n+1
        if not found:
            cross_center.append([x,y,1])
    n=0
    for p in cross_center:
        print("%d (%4d,%4d)" %(n+1,int(p[0]),int(p[1])))
        cv2.rectangle(org, (int(p[0]) - tw//2, int(p[1]) - th//2), (int(p[0]) + tw//2, int(p[1]) + th//2), (255,0,0), 1)
        cv2.circle(org, (int(p[0]),int(p[1])), 1, (0, 0, 255), -1) #red center point of cross
        n=n+1
    cv2.imshow("img", org)
    cv2.waitKey(0)

cv2.destroyAllWindows()
# Result
#img etch.jpg
#1 ( 151, 692)
#2 (1152, 747)
#img etch1.jpg
#1 ( 152, 694)
#2 (1159, 698)
#img etch2.jpg
#1 ( 152, 694)
#2 (1159, 698)
#img etch3.jpg
#1 (1154, 693)
#2 ( 149, 693)
