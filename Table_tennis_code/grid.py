# import required library
import cv2

# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
   
   # checking for left mouse clicks
   if event == cv2.EVENT_LBUTTONDOWN:
      print('Left Click')
      print(f'({x},{y})')
 
   # put coordinates as text on the image
   cv2.putText(img, f'({x},{y})', (x, y),   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
   cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
   if event == cv2.EVENT_RBUTTONDOWN:
      print('Right Click')
      print(f'({x},{y})')
 
      # put coordinates as text on the image
      cv2.putText(img, f'({x},{y})', (x, y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
      cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# read the input image
img = cv2.imread("C:/Users/HP/Desktop/Table_tennis/IMG_20230829_130126.jpg")

# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
   cv2.imshow('Point Coordinates', img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()